import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../controllers/vision_test_controller.dart';
import '../core/models/evidence.dart';
import '../services/vision/camera_controller_service.dart';
import '../services/vision/efficientdet_model.dart';
import '../services/vision/movinet_stream_model.dart';
import '../services/vision/object_tracker.dart';
import '../services/vision/behavioral_feature_extractor.dart';
import '../services/vision/visual_evidence_processor.dart';
import '../core/models/test_config.dart';
import '../widgets/detection_overlay_painter.dart';
import '../widgets/evidence_json_panel.dart';
import '../widgets/perf_stats_bar.dart';

class VisionTestScreen extends StatefulWidget {
  final EfficientDetModel efficientDet;
  final MovinetStreamModel movinet;
  const VisionTestScreen({super.key, required this.efficientDet, required this.movinet});

  @override
  State<VisionTestScreen> createState() => _VisionTestScreenState();
}

class _VisionTestScreenState extends State<VisionTestScreen> {
  late final VisionTestController controller;
  final CameraControllerService cameraService = CameraControllerService();

  List<Detection> _detections = [];
  List<TrackedObject> _tracks = [];
  List<Evidence> _tierAEvidence = [];
  List<ActionClassResult> _actionResults = [];
  Evidence? _tierBEvidence;
  String _status = 'Idle';

  bool _tierAEnabled = true;
  bool _tierBEnabled = false;
  double _simulatedBattery = 100;
  double _simulatedHeading = 0;
  double _simulatedSpeed = 0;

  @override
  void initState() {
    super.initState();
    // AUDIT FIX (T-15): explicitly source these caps from TesterConfig
    // instead of relying on each class's own unrelated hardcoded default —
    // the two happened to numerically match today, masking the fact that
    // tuning TesterConfig previously had no effect here at all.
    const cfg = TesterConfig.defaults;
    controller = VisionTestController(
      camera: cameraService,
      detector: widget.efficientDet,
      tracker: ObjectTracker(),
      heuristics: BehavioralFeatureExtractor(
          behaviorHeuristicConfidenceCap: cfg.behaviorHeuristicConfidenceCap),
      movinet: widget.movinet,
      processor: VisualEvidenceProcessor(actionProxyConfidenceCap: cfg.actionProxyConfidenceCap),
      config: cfg,
    );
    controller.detectionsStream.listen((d) => setState(() => _detections = d));
    controller.tracksStream.listen((t) => setState(() => _tracks = t));
    controller.tierAEvidenceStream.listen((e) => setState(() => _tierAEvidence = e));
    controller.actionResultsStream.listen((r) => setState(() => _actionResults = r));
    controller.tierBEvidenceStream.listen((e) => setState(() => _tierBEvidence = e));
    controller.statusStream.listen((s) => setState(() => _status = s));
  }

  Future<void> _toggleRunning() async {
    if (controller.isRunning) {
      await controller.stop();
    } else {
      controller.setTierA(_tierAEnabled);
      controller.setTierB(_tierBEnabled);
      controller.simulatedBatteryPercent = _simulatedBattery;
      controller.simulatedUserHeadingDegrees = _simulatedHeading;
      controller.simulatedUserSpeedMps = _simulatedSpeed;
      await controller.start();
    }
    setState(() {});
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vision Test — EfficientDet + MoViNet')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: CheckboxListTile(
                    title: const Text('Tier A (EfficientDet + heuristics)'),
                    value: _tierAEnabled,
                    onChanged: controller.isRunning
                        ? null
                        : (v) => setState(() => _tierAEnabled = v ?? false),
                    dense: true,
                  ),
                ),
              ],
            ),
            Row(
              children: [
                Expanded(
                  child: CheckboxListTile(
                    title: const Text('Tier B (MoViNet-Stream, Option 2)'),
                    value: _tierBEnabled,
                    onChanged: controller.isRunning
                        ? null
                        : (v) => setState(() => _tierBEnabled = v ?? false),
                    dense: true,
                  ),
                ),
              ],
            ),
            // AUDIT FIX (T-3/T-5): exposes the reset-cadence choice
            // explicitly, since it materially changes whether observed
            // heuristic evidence (e.g. possible_following) will reproduce
            // once integrated into production.
            CheckboxListTile(
              title: const Text('Reset tracker/heuristics each cycle (matches production)'),
              subtitle: const Text(
                  'Off = cumulative testing mode: state persists across cycles, '
                  'which has NO production equivalent.'),
              value: controller.matchProductionCadence,
              onChanged: controller.isRunning
                  ? null
                  : (v) => setState(() => controller.matchProductionCadence = v ?? true),
              dense: true,
            ),
            const SizedBox(height: 8),
            Text('Simulated Battery: ${_simulatedBattery.toStringAsFixed(0)}% '
                '(Tier B gate at 40%)'),
            Slider(
              value: _simulatedBattery,
              min: 0,
              max: 100,
              divisions: 20,
              label: _simulatedBattery.toStringAsFixed(0),
              onChanged: (v) {
                setState(() => _simulatedBattery = v);
                controller.simulatedBatteryPercent = v;
              },
            ),
            Text('Simulated Heading: ${_simulatedHeading.toStringAsFixed(0)}° '
                '(for possible_following turn-correlation test)'),
            Slider(
              value: _simulatedHeading,
              min: 0,
              max: 359,
              onChanged: (v) {
                setState(() => _simulatedHeading = v);
                controller.simulatedUserHeadingDegrees = v;
              },
            ),
            Text('Simulated Speed: ${_simulatedSpeed.toStringAsFixed(1)} m/s '
                '(for possible_blocked_path speed-drop test)'),
            Slider(
              value: _simulatedSpeed,
              min: 0,
              max: 3,
              onChanged: (v) {
                setState(() => _simulatedSpeed = v);
                controller.simulatedUserSpeedMps = v;
              },
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(child: Text('Status: $_status')),
                ElevatedButton(
                  onPressed: _toggleRunning,
                  style: ElevatedButton.styleFrom(
                      backgroundColor: controller.isRunning ? Colors.red : Colors.green),
                  child: Text(controller.isRunning ? 'Stop' : 'Start'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (cameraService.isActive && cameraService.rawController != null)
              AspectRatio(
                aspectRatio: cameraService.rawController!.value.aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(cameraService.rawController!),
                    CustomPaint(
                      painter: DetectionOverlayPainter(detections: _detections, tracks: _tracks),
                    ),
                  ],
                ),
              )
            else
              Container(
                height: 200,
                color: Colors.black12,
                alignment: Alignment.center,
                child: const Text('Camera preview appears when running'),
              ),
            const SizedBox(height: 12),
            PerfStatsBar(label: 'Tier A (EfficientDet)', stats: controller.tierAPerf),
            const SizedBox(height: 6),
            PerfStatsBar(label: 'Tier B (MoViNet)', stats: controller.tierBPerf),
            const SizedBox(height: 16),
            if (_actionResults.isNotEmpty) ...[
              const Text('MoViNet Top Actions (raw model output)',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              ..._actionResults.map((r) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      children: [
                        SizedBox(width: 160, child: Text(r.label, overflow: TextOverflow.ellipsis)),
                        Expanded(child: LinearProgressIndicator(value: r.score.clamp(0.0, 1.0))),
                        SizedBox(width: 50, child: Text(' ${(r.score * 100).toStringAsFixed(1)}%')),
                      ],
                    ),
                  )),
              const SizedBox(height: 16),
            ],
            EvidenceJsonPanel(
              title: 'Tier A Evidence (detections + heuristics)',
              evidenceList: _tierAEvidence,
              accentColor: Colors.teal,
            ),
            const SizedBox(height: 12),
            EvidenceJsonPanel(
              title: 'Tier B Evidence (MoViNet action proxy — Option 2)',
              evidenceList: _tierBEvidence == null ? [] : [_tierBEvidence!],
              accentColor: Colors.deepPurple,
            ),
          ],
        ),
      ),
    );
  }
}