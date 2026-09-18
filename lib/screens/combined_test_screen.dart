import 'package:flutter/material.dart';
import '../controllers/audio_test_controller.dart';
import '../controllers/vision_test_controller.dart';
import '../controllers/combined_test_controller.dart';
import '../core/models/evidence.dart';
import '../services/audio/yamnet_model.dart';
import '../services/audio/audio_evidence_processor.dart';
import '../services/audio/mic_capture_service.dart';
import '../services/vision/camera_controller_service.dart';
import '../services/vision/efficientdet_model.dart';
import '../services/vision/movinet_stream_model.dart';
import '../services/vision/object_tracker.dart';
import '../services/vision/behavioral_feature_extractor.dart';
import '../services/vision/visual_evidence_processor.dart';
import '../core/models/test_config.dart';
import '../widgets/evidence_json_panel.dart';

class CombinedTestScreen extends StatefulWidget {
  final YamnetModel yamnet;
  final EfficientDetModel efficientDet;
  final MovinetStreamModel movinet;

  const CombinedTestScreen({
    super.key,
    required this.yamnet,
    required this.efficientDet,
    required this.movinet,
  });

  @override
  State<CombinedTestScreen> createState() => _CombinedTestScreenState();
}

class _CombinedTestScreenState extends State<CombinedTestScreen> {
  late final AudioTestController audioController;
  late final VisionTestController visionController;
  late final CombinedTestController combinedController;

  List<Evidence> _mergedFeed = [];
  String _status = 'Idle';

  bool _includeAudio = true;
  bool _includeVisionA = true;
  bool _includeVisionB = false;

  @override
  void initState() {
    super.initState();
    audioController = AudioTestController(
      model: widget.yamnet,
      processor: AudioEvidenceProcessor(),
      micCapture: MicCaptureService(),
    );
    // AUDIT FIX (T-15): same explicit config-sourcing fix as the vision screen.
    const cfg = TesterConfig.defaults;
    visionController = VisionTestController(
      camera: CameraControllerService(),
      detector: widget.efficientDet,
      tracker: ObjectTracker(),
      heuristics: BehavioralFeatureExtractor(
          behaviorHeuristicConfidenceCap: cfg.behaviorHeuristicConfidenceCap),
      movinet: widget.movinet,
      processor: VisualEvidenceProcessor(actionProxyConfidenceCap: cfg.actionProxyConfidenceCap),
      config: cfg,
    );
    combinedController = CombinedTestController(
      audioController: audioController,
      visionController: visionController,
    );
    combinedController.mergedFeedStream.listen((feed) => setState(() => _mergedFeed = feed));
    combinedController.statusStream.listen((s) => setState(() => _status = s));
  }

  Future<void> _toggle() async {
    if (combinedController.isRunning) {
      await combinedController.stop();
    } else {
      await combinedController.start(
        includeAudio: _includeAudio,
        includeVisionTierA: _includeVisionA,
        includeVisionTierB: _includeVisionB,
      );
    }
    setState(() {});
  }

  @override
  void dispose() {
    combinedController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Combined Test — Audio + Vision Concurrently')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Runs audio and vision on independent timers, exactly as the '
              'real orchestrator would activate them separately. Confirms '
              'neither modality starves the other and evidence timestamps '
              'interleave sensibly.\n\n'
              // AUDIT FIX (T-18): explicit caveat — this mode does NOT
              // simulate AdaptiveSensingPolicy's risk/confidence/battery-
              // gated activation timing. It only validates that the models
              // run correctly at the same time, not real-world cadence.
              'NOTE: this does not simulate the real risk/confidence/battery-'
              'gated activation timing — it only confirms both models can '
              'run concurrently without interfering with each other.',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            const SizedBox(height: 12),
            CheckboxListTile(
              title: const Text('Include Audio (YAMNet)'),
              value: _includeAudio,
              onChanged: combinedController.isRunning
                  ? null
                  : (v) => setState(() => _includeAudio = v ?? false),
              dense: true,
            ),
            CheckboxListTile(
              title: const Text('Include Vision Tier A (EfficientDet)'),
              value: _includeVisionA,
              onChanged: combinedController.isRunning
                  ? null
                  : (v) => setState(() => _includeVisionA = v ?? false),
              dense: true,
            ),
            CheckboxListTile(
              title: const Text('Include Vision Tier B (MoViNet, Option 2)'),
              value: _includeVisionB,
              onChanged: combinedController.isRunning
                  ? null
                  : (v) => setState(() => _includeVisionB = v ?? false),
              dense: true,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(child: Text('Status: $_status')),
                ElevatedButton(
                  onPressed: _toggle,
                  style: ElevatedButton.styleFrom(
                      backgroundColor: combinedController.isRunning ? Colors.red : Colors.green),
                  child: Text(combinedController.isRunning ? 'Stop' : 'Start'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(
              child: EvidenceJsonPanel(
                title: 'Merged, Timestamp-Sorted Evidence Feed (last 100)',
                evidenceList: _mergedFeed,
                accentColor: Colors.indigo,
              ),
            ),
          ],
        ),
      ),
    );
  }
}