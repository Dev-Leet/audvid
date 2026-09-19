import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../controllers/vision_cd_test_controller.dart';
import '../core/models/evidence.dart';
import '../services/vision/camera_controller_service.dart';
import '../services/vision/face_detection_service.dart';
import '../services/vision/pose_detection_service.dart';
import '../services/vision/face_evidence_processor.dart';
import '../services/vision/pose_evidence_processor.dart';
import '../widgets/mlkit_overlay_painter.dart';
import '../widgets/evidence_json_panel.dart';
import '../widgets/perf_stats_bar.dart';

class VisionCdTestScreen extends StatefulWidget {
  final FaceDetectionService faceService;
  final PoseDetectionService poseService;
  
  const VisionCdTestScreen({super.key, required this.faceService, required this.poseService});

  @override
  State<VisionCdTestScreen> createState() => _VisionCdTestScreenState();
}

class _VisionCdTestScreenState extends State<VisionCdTestScreen> {
  late final VisionCdTestController controller;
  final CameraControllerService cameraService = CameraControllerService();

  List<Face> _faces = [];
  Pose? _pose;
  Evidence? _tierCEvidence;
  Evidence? _tierDEvidence;
  String _status = 'Idle';

  bool _tierCEnabled = true;
  bool _tierDEnabled = false;

  @override
  void initState() {
    super.initState();
    controller = VisionCdTestController(
      camera: cameraService,
      faceService: widget.faceService,
      poseService: widget.poseService,
      faceProcessor: FaceEvidenceProcessor(),
      poseProcessor: PoseEvidenceProcessor(),
    );
    controller.facesStream.listen((f) { if (mounted) setState(() => _faces = f); });
    controller.poseStream.listen((p) { if (mounted) setState(() => _pose = p); });
    controller.tierCEvidenceStream.listen((e) { if (mounted) setState(() => _tierCEvidence = e); });
    controller.tierDEvidenceStream.listen((e) { if (mounted) setState(() => _tierDEvidence = e); });
    controller.statusStream.listen((s) { if (mounted) setState(() => _status = s); });
  }

  Future<void> _toggleRunning() async {
    if (controller.isRunning) {
      await controller.stop();
    } else {
      controller.setTierC(_tierCEnabled);
      controller.setTierD(_tierDEnabled);
      await controller.start();
    }
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vision Test — Face & Pose ONLY')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            CheckboxListTile(
              title: const Text('Tier C (Face Detection)'),
              subtitle: const Text('Assigns session ID, tracks head angle. No biometrics.'),
              value: _tierCEnabled,
              onChanged: controller.isRunning
                  ? null
                  : (v) => setState(() => _tierCEnabled = v ?? false),
              dense: true,
            ),
            CheckboxListTile(
              title: const Text('Tier D (Pose/Skeleton Detection)'),
              subtitle: const Text('Single-person full body skeleton tracking.'),
              value: _tierDEnabled,
              onChanged: controller.isRunning
                  ? null
                  : (v) => setState(() => _tierDEnabled = v ?? false),
              dense: true,
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
              Container(
                height: MediaQuery.of(context).size.height * 0.5, // Make preview taller
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade800, width: 2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: InteractiveViewer(
                    panEnabled: true,
                    minScale: 1.0,
                    maxScale: 5.0,
                    child: Center(
                      child: AspectRatio(
                        aspectRatio: cameraService.rawController!.value.aspectRatio,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            CameraPreview(cameraService.rawController!),
                            CustomPaint(
                              painter: MlkitOverlayPainter(
                                faces: _faces, 
                                pose: _pose,
                                imageSize: controller.imageSize,
                                rotation: controller.imageRotation,
                                lensDirection: controller.lensDirection,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
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
            PerfStatsBar(label: 'Tier C (Face)', stats: controller.tierCPerf),
            const SizedBox(height: 6),
            PerfStatsBar(label: 'Tier D (Pose)', stats: controller.tierDPerf),
            const SizedBox(height: 16),
            EvidenceJsonPanel(
              title: 'Tier C Evidence (Face)',
              evidenceList: _tierCEvidence == null ? [] : [_tierCEvidence!],
              accentColor: Colors.blueAccent,
            ),
            const SizedBox(height: 12),
            EvidenceJsonPanel(
              title: 'Tier D Evidence (Pose)',
              evidenceList: _tierDEvidence == null ? [] : [_tierDEvidence!],
              accentColor: Colors.orangeAccent,
            ),
          ],
        ),
      ),
    );
  }
}