import 'dart:async';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../core/models/evidence.dart';
import '../core/utils/logger.dart';
import '../services/vision/camera_controller_service.dart';
import '../services/vision/mlkit_input_image_converter.dart';
import '../services/vision/face_detection_service.dart';
import '../services/vision/pose_detection_service.dart';
import '../services/vision/face_evidence_processor.dart';
import '../services/vision/pose_evidence_processor.dart';
import 'perf_stats.dart';

class VisionCdTestController {
  final CameraControllerService camera;
  final FaceDetectionService faceService;
  final PoseDetectionService poseService;
  final FaceEvidenceProcessor faceProcessor;
  final PoseEvidenceProcessor poseProcessor;

  final PerfStats tierCPerf = PerfStats();
  final PerfStats tierDPerf = PerfStats();

  final _tierCEvidenceController = StreamController<Evidence>.broadcast();
  final _tierDEvidenceController = StreamController<Evidence>.broadcast();
  final _statusController = StreamController<String>.broadcast();
  
  // Expose the raw outputs for overlay drawing
  final _facesController = StreamController<List<Face>>.broadcast();
  final _poseController = StreamController<Pose?>.broadcast();

  Stream<Evidence> get tierCEvidenceStream => _tierCEvidenceController.stream;
  Stream<Evidence> get tierDEvidenceStream => _tierDEvidenceController.stream;
  Stream<String> get statusStream => _statusController.stream;
  Stream<List<Face>> get facesStream => _facesController.stream;
  Stream<Pose?> get poseStream => _poseController.stream;

  bool _tierCRequested = false;
  bool _tierDRequested = false;
  bool _effectiveTierC = false;
  bool _effectiveTierD = false;
  
  bool _isRunning = false;
  bool get isRunning => _isRunning;
  bool get tierCEnabled => _tierCRequested;
  bool get tierDEnabled => _tierDRequested;
  
  // Matches the existing test cadence of ~200ms per frame
  final Duration cycleInterval = const Duration(milliseconds: 200);

  bool _stopRequested = false;

  VisionCdTestController({
    required this.camera,
    required this.faceService,
    required this.poseService,
    required this.faceProcessor,
    required this.poseProcessor,
  });

  void setTierC(bool enabled) => _tierCRequested = enabled;
  void setTierD(bool enabled) => _tierDRequested = enabled;

  Future<void> start() async {
    if (_isRunning) return;
    if (!_tierCRequested && !_tierDRequested) {
      _statusController.add('Enable at least one tier before starting.');
      return;
    }

    _effectiveTierC = _tierCRequested && faceService.isLoaded;
    _effectiveTierD = _tierDRequested && poseService.isLoaded;

    if (_tierCRequested && !faceService.isLoaded) {
      _statusController.add('Tier C requested but ML Kit Face is not loaded — skipping Tier C.');
    }
    if (_tierDRequested && !poseService.isLoaded) {
      _statusController.add('Tier D requested but ML Kit Pose is not loaded — skipping Tier D.');
    }
    if (!_effectiveTierC && !_effectiveTierD) {
      _statusController.add('No requested tier has a loaded model — cannot start.');
      return;
    }

    final activated = await camera.activate(resolution: ResolutionPreset.max);
    if (!activated) {
      _statusController.add('Camera activation failed — check permission/hardware.');
      return;
    }

    _isRunning = true;
    _stopRequested = false;
    _statusController.add('Running (Tier C: $_effectiveTierC, Tier D: $_effectiveTierD)');
    unawaited(_runLoop());
  }

  Future<void> stop() async {
    _stopRequested = true;
    _isRunning = false;
    await camera.deactivate();
    _statusController.add('Stopped');
  }

  Future<void> _runLoop() async {
    while (!_stopRequested) {
      await _runCycle();
      if (_stopRequested) break;
      await Future.delayed(cycleInterval);
    }
  }

  Future<void> _runCycle() async {
    final frame = await camera
        .waitForNextFrame(timeout: const Duration(seconds: 2))
        .timeout(const Duration(seconds: 3), onTimeout: () => null);
        
    if (frame == null) {
       // Stalled camera
       if (_effectiveTierC) {
          _tierCEvidenceController.add(Evidence(
            modality: Modality.vision,
            timestamp: DateTime.now(),
            confidence: 0,
            riskContribution: 0,
            isUnavailable: true,
            payload: {'status': 'no_frames_processed'},
          ));
          tierCPerf.recordFailure();
       }
       if (_effectiveTierD) {
          _tierDEvidenceController.add(Evidence(
            modality: Modality.vision,
            timestamp: DateTime.now(),
            confidence: 0,
            riskContribution: 0,
            isUnavailable: true,
            payload: {'status': 'no_frames_processed'},
          ));
          tierDPerf.recordFailure();
       }
       return;
    }

    final inputImage = MlkitInputImageConverter.convert(
      frame,
      sensorOrientation: camera.rawController!.description.sensorOrientation,
      isFrontCamera: camera.rawController!.description.lensDirection == CameraLensDirection.front,
      deviceOrientation: camera.rawController!.value.deviceOrientation,
    );

    if (inputImage == null) {
      _statusController.add('Failed to convert camera frame to InputImage.');
      return;
    }

    if (_effectiveTierC) {
      final cStart = DateTime.now();
      try {
        final faces = await faceService.detect(inputImage);
        _facesController.add(faces);
        final evidence = faceProcessor.process(faces, DateTime.now(), frame.width, frame.height);
        _tierCEvidenceController.add(evidence);
        tierCPerf.recordLatency(DateTime.now().difference(cStart).inMilliseconds);
      } catch (e) {
        tierCPerf.recordFailure();
        logger.error('VisionCdTestController', 'Tier C cycle failed', e);
      }
    }

    if (_effectiveTierD) {
      final dStart = DateTime.now();
      try {
        final pose = await poseService.detect(inputImage);
        _poseController.add(pose);
        final evidence = poseProcessor.process(pose, DateTime.now(), frame.width, frame.height);
        _tierDEvidenceController.add(evidence);
        tierDPerf.recordLatency(DateTime.now().difference(dStart).inMilliseconds);
      } catch (e) {
        tierDPerf.recordFailure();
        logger.error('VisionCdTestController', 'Tier D cycle failed', e);
      }
    }
  }

  Future<void> dispose() async {
    await stop();
    await _tierCEvidenceController.close();
    await _tierDEvidenceController.close();
    await _statusController.close();
    await _facesController.close();
    await _poseController.close();
  }
}

void unawaited(Future<void> future) {}
