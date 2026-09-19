import 'dart:async';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:camera/camera.dart' show CameraLensDirection;
import '../core/models/evidence.dart';
import '../core/models/test_config.dart';
import '../core/utils/logger.dart';
import '../services/vision/camera_controller_service.dart';
import '../services/vision/vision_preprocessing.dart';
import '../services/vision/efficientdet_model.dart';
import '../services/vision/object_tracker.dart';
import '../services/vision/behavioral_feature_extractor.dart';
import '../services/vision/movinet_stream_model.dart';
import '../services/vision/visual_evidence_processor.dart';
import 'perf_stats.dart';

class VisionTestController {
  final CameraControllerService camera;
  final EfficientDetModel detector;
  final ObjectTracker tracker;
  final BehavioralFeatureExtractor heuristics;
  final MovinetStreamModel movinet;
  final VisualEvidenceProcessor processor;
  final TesterConfig config;

  final PerfStats tierAPerf = PerfStats();
  final PerfStats tierBPerf = PerfStats();

  final _detectionsController = StreamController<List<Detection>>.broadcast();
  final _tracksController = StreamController<List<TrackedObject>>.broadcast();
  final _tierAEvidenceController = StreamController<List<Evidence>>.broadcast();
  final _actionResultsController = StreamController<List<ActionClassResult>>.broadcast();
  final _tierBEvidenceController = StreamController<Evidence>.broadcast();
  final _statusController = StreamController<String>.broadcast();

  Stream<List<Detection>> get detectionsStream => _detectionsController.stream;
  Stream<List<TrackedObject>> get tracksStream => _tracksController.stream;
  Stream<List<Evidence>> get tierAEvidenceStream => _tierAEvidenceController.stream;
  Stream<List<ActionClassResult>> get actionResultsStream => _actionResultsController.stream;
  Stream<Evidence> get tierBEvidenceStream => _tierBEvidenceController.stream;
  Stream<String> get statusStream => _statusController.stream;

  bool _tierARequested = false;
  bool _tierBRequested = false;
  bool _effectiveTierA = false; // AUDIT FIX (T-8): requested AND model loaded
  bool _effectiveTierB = false;
  bool _isRunning = false;
  bool get isRunning => _isRunning;
  bool get tierAEnabled => _tierARequested;
  bool get tierBEnabled => _tierBRequested;

  /// AUDIT FIX (T-3/T-5): when true (default), tracker/heuristic state is
  /// reset at the START of every Tier A cycle — matching the real
  /// orchestrator's per-activation semantics exactly (each Tier A
  /// activation in production starts fresh). Set to false only if you
  /// deliberately want to observe cumulative/long-run heuristic behavior
  /// that has NO production equivalent — the UI labels this explicitly.
  bool matchProductionCadence = true;

  double simulatedBatteryPercent = 100;
  double? simulatedUserHeadingDegrees;
  double? simulatedUserSpeedMps;

  bool _stopRequested = false;

  VisionTestController({
    required this.camera,
    required this.detector,
    required this.tracker,
    required this.heuristics,
    required this.movinet,
    required this.processor,
    this.config = TesterConfig.defaults,
  });

  void setTierA(bool enabled) => _tierARequested = enabled;
  void setTierB(bool enabled) => _tierBRequested = enabled;

  Future<void> start() async {
    if (_isRunning) return;
    if (!_tierARequested && !_tierBRequested) {
      _statusController.add('Enable at least one tier before starting.');
      return;
    }

    // AUDIT FIX (T-8): degrade PER TIER instead of refusing to start
    // entirely when only one requested tier's model failed to load. This
    // app's whole purpose is testing each model — one failed model must
    // not block testing of the other, working one.
    _effectiveTierA = _tierARequested && detector.isLoaded;
    _effectiveTierB = _tierBRequested && movinet.isLoaded;

    if (_tierARequested && !detector.isLoaded) {
      _statusController.add('Tier A requested but EfficientDet-Lite0 is not loaded — skipping Tier A.');
    }
    if (_tierBRequested && !movinet.isLoaded) {
      _statusController.add('Tier B requested but MoViNet-Stream is not loaded — skipping Tier B.');
    }
    if (!_effectiveTierA && !_effectiveTierB) {
      _statusController.add('No requested tier has a loaded model — cannot start.');
      return;
    }

    final activated = await camera.activate();
    if (!activated) {
      _statusController.add('Camera activation failed — check permission/hardware.');
      return;
    }

    _isRunning = true;
    _stopRequested = false;
    _statusController.add('Running (Tier A: $_effectiveTierA, Tier B: $_effectiveTierB)');
    unawaited(_runLoop());
  }

  Future<void> stop() async {
    _stopRequested = true;
    _isRunning = false;
    await camera.deactivate();
    tracker.reset();
    heuristics.reset();
    if (movinet.isLoaded) movinet.resetStream();
    _statusController.add('Stopped');
  }

  Future<void> _runLoop() async {
    while (!_stopRequested) {
      if (_effectiveTierA) await _runTierACycle();
      if (_effectiveTierB) {
        if (simulatedBatteryPercent >= config.minBatteryForVisionTierB) {
          await _runTierBCycle();
        } else {
          _statusController.add(
              'Tier B skipped: simulated battery (${simulatedBatteryPercent.toStringAsFixed(0)}%) '
              'below gate (${config.minBatteryForVisionTierB.toStringAsFixed(0)}%)');
        }
      }
      if (_stopRequested) break;
      await Future.delayed(const Duration(milliseconds: 500));
    }
  }

  Future<void> _runTierACycle() async {
    // AUDIT FIX (T-3/T-5): reset at the START of the cycle by default,
    // mirroring the real VisionController.runTierA's `finally` reset after
    // every activation. Without this, heading/persistence-based evidence
    // (e.g. `confirmed_by_turn`) accumulates over an arbitrarily long
    // tester session with no production equivalent.
    if (matchProductionCadence) {
      tracker.reset();
      heuristics.reset();
    }

    final cycleStart = DateTime.now();
    try {
      final aggregated = <String, Evidence>{};
      List<Detection> lastDetections = const [];
      var framesProcessed = 0; // AUDIT FIX (T-4/T-7)

      for (var i = 0; i < config.visionBurstFrameCount && !_stopRequested; i++) {
        final frame = await camera
            .waitForNextFrame(timeout: const Duration(seconds: 2))
            .timeout(const Duration(seconds: 3), onTimeout: () => null);
        if (frame == null) break;

        final frameHeightPx = camera.previewHeightPx ?? frame.height;
        
        // Calculate dynamic rotation angle for Tier A
        int deviceOrientationDegrees = 0;
        switch (camera.rawController!.value.deviceOrientation) {
          case DeviceOrientation.portraitUp:
            deviceOrientationDegrees = 0;
            break;
          case DeviceOrientation.landscapeLeft:
            deviceOrientationDegrees = 90;
            break;
          case DeviceOrientation.portraitDown:
            deviceOrientationDegrees = 180;
            break;
          case DeviceOrientation.landscapeRight:
            deviceOrientationDegrees = 270;
            break;
        }
        
        final sensorOrientation = camera.rawController!.description.sensorOrientation;
        final isFrontCamera = camera.rawController!.description.lensDirection == CameraLensDirection.front;
        final rotationCompensation = isFrontCamera
            ? (sensorOrientation + deviceOrientationDegrees) % 360
            : (sensorOrientation - deviceOrientationDegrees + 360) % 360;

        final rgb = await VisionPreprocessing.cameraImageToRgbBytesIsolate(
            frame, VisionPreprocessing.efficientDetInputSize, rotationDegrees: rotationCompensation);
        lastDetections = detector.infer(rgb);
        _detectionsController.add(lastDetections);
        framesProcessed++;

        final tracked = tracker.update(
          lastDetections.map((d) => (label: d.label, box: d.boundingBox)).toList(),
        );
        _tracksController.add(tracked);

        final heuristicEvidence = heuristics.extract(
          tracks: tracked,
          cameraFocalLengthPx: 1400.0, // placeholder — see C-12 note in main app
          frameHeightPx: frameHeightPx,
          cameraHorizontalFovDegrees: 60.0,
          userHeadingDegrees: simulatedUserHeadingDegrees,
          userSpeedMetersPerSecond: simulatedUserSpeedMps,
          timestamp: DateTime.now(),
        );
        for (final e in heuristicEvidence) {
          aggregated[e.sourceKey ?? '${e.modality.name}_$i'] = e;
        }

        await Future.delayed(Duration(milliseconds: config.visionFrameSampleIntervalMs));
      }

      // AUDIT FIX (T-4): previously `processDetections([], ...)` was
      // called unconditionally even when zero frames were obtained,
      // producing an ordinary "safe" evidence item indistinguishable from
      // a genuinely working camera confirming nobody is around. Now a
      // stalled camera produces an explicit unavailable evidence item.
      if (framesProcessed == 0) {
        _tierAEvidenceController.add([
          Evidence(
            modality: Modality.vision,
            timestamp: DateTime.now(),
            confidence: 0,
            riskContribution: 0,
            isUnavailable: true,
            payload: {'status': 'no_frames_processed'},
          ),
        ]);
        tierAPerf.recordFailure(); // AUDIT FIX (T-7)
        return;
      }

      final detectionEvidence = processor.processDetections(lastDetections, DateTime.now());
      final allEvidence = [detectionEvidence, ...aggregated.values];
      _tierAEvidenceController.add(allEvidence);

      tierAPerf.recordLatency(DateTime.now().difference(cycleStart).inMilliseconds);
    } catch (e) {
      logger.error('VisionTestController', 'Tier A cycle failed', e);
      tierAPerf.recordFailure();
      _statusController.add('Tier A error: $e');
    }
  }

  Future<void> _runTierBCycle() async {
    final cycleStart = DateTime.now();
    try {
      movinet.resetStream();
      List<ActionClassResult> lastResults = const [];
      var framesProcessed = 0;

      for (var i = 0; i < config.movinetWindowFrames && !_stopRequested; i++) {
        if (simulatedBatteryPercent < config.minBatteryForVisionTierB) {
          _statusController.add('Tier B aborted mid-window: simulated battery dropped below gate');
          break;
        }

        final frame = await camera
            .waitForNextFrame(timeout: const Duration(seconds: 2))
            .timeout(const Duration(seconds: 3), onTimeout: () => null);
        if (frame == null) break;

        // Calculate dynamic rotation angle for Tier B
        int deviceOrientationDegrees = 0;
        switch (camera.rawController!.value.deviceOrientation) {
          case DeviceOrientation.portraitUp:
            deviceOrientationDegrees = 0;
            break;
          case DeviceOrientation.landscapeLeft:
            deviceOrientationDegrees = 90;
            break;
          case DeviceOrientation.portraitDown:
            deviceOrientationDegrees = 180;
            break;
          case DeviceOrientation.landscapeRight:
            deviceOrientationDegrees = 270;
            break;
        }
        
        final sensorOrientation = camera.rawController!.description.sensorOrientation;
        final isFrontCamera = camera.rawController!.description.lensDirection == CameraLensDirection.front;
        final rotationCompensation = isFrontCamera
            ? (sensorOrientation + deviceOrientationDegrees) % 360
            : (sensorOrientation - deviceOrientationDegrees + 360) % 360;

        final rgb = await VisionPreprocessing.cameraImageToRgbBytesIsolate(
            frame, MovinetStreamModel.inputFrameSize, rotationDegrees: rotationCompensation);
        lastResults = movinet.processFrame(rgb);
        _actionResultsController.add(lastResults);
        framesProcessed++;

        await Future.delayed(Duration(milliseconds: config.visionFrameSampleIntervalMs));
      }

      // AUDIT FIX (T-6): previously, a zero-frame cycle emitted NO evidence
      // at all, leaving the JSON panel showing a stale previous result
      // with no way to tell "still confidently benign" apart from "hasn't
      // actually run in several cycles". Now always emits per cycle.
      if (framesProcessed == 0) {
        _tierBEvidenceController.add(Evidence(
          modality: Modality.visionAction,
          timestamp: DateTime.now(),
          confidence: 0,
          riskContribution: 0,
          isUnavailable: true,
          payload: {'status': 'no_frames_processed'},
        ));
        tierBPerf.recordFailure(); // AUDIT FIX (T-7)
        return;
      }

      final evidence = processor.processAction(lastResults, DateTime.now());
      _tierBEvidenceController.add(evidence);
      tierBPerf.recordLatency(DateTime.now().difference(cycleStart).inMilliseconds);
    } catch (e) {
      logger.error('VisionTestController', 'Tier B cycle failed', e);
      tierBPerf.recordFailure();
      _statusController.add('Tier B error: $e');
    }
  }

  /// AUDIT FIX (T-2) — THE CRITICAL FIX: `detector`/`movinet` are SHARED
  /// SINGLETON models created once in main.dart and reused across every
  /// screen (Vision Test AND Combined Test). The previous version called
  /// `detector.dispose()`/`movinet.dispose()` here, permanently destroying
  /// them the first time a user navigated away from Vision Test — silently
  /// breaking vision testing for the rest of the app session. This
  /// controller does not own those models and must never dispose them;
  /// model lifecycle belongs solely to whoever created them (main.dart's
  /// bootstrap screen).
  Future<void> dispose() async {
    await stop();
    await _detectionsController.close();
    await _tracksController.close();
    await _tierAEvidenceController.close();
    await _actionResultsController.close();
    await _tierBEvidenceController.close();
    await _statusController.close();
  }
}

void unawaited(Future<void> future) {}