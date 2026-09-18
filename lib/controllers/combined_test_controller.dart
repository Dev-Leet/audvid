import 'dart:async';
import '../core/models/evidence.dart';
import '../core/models/test_config.dart';
import '../core/utils/logger.dart';
import 'audio_test_controller.dart';
import 'vision_test_controller.dart';

/// Runs AudioTestController and VisionTestController CONCURRENTLY on
/// independent timers — not lockstep — mirroring how the real orchestrator
/// activates audio and vision sensing as separate, independently-timed
/// decisions rather than one synchronized cycle. Merges every Evidence
/// object from both into a single timestamp-ordered feed so you can
/// visually confirm neither modality starves the other and that
/// timestamps interleave the way the real Fusion Engine would see them.
class CombinedTestController {
  final AudioTestController audioController;
  final VisionTestController visionController;
  final TesterConfig config;

  final List<Evidence> _mergedLog = [];
  final _mergedFeedController = StreamController<List<Evidence>>.broadcast();
  final _statusController = StreamController<String>.broadcast();

  Stream<List<Evidence>> get mergedFeedStream => _mergedFeedController.stream;
  Stream<String> get statusStream => _statusController.stream;

  StreamSubscription? _audioSub;
  StreamSubscription? _visionASub;
  StreamSubscription? _visionBSub;

  bool _isRunning = false;
  bool get isRunning => _isRunning;

  static const int _maxLogEntries = 100;

  CombinedTestController({
    required this.audioController,
    required this.visionController,
    this.config = TesterConfig.defaults,
  });

  Future<void> start({required bool includeAudio, required bool includeVisionTierA, required bool includeVisionTierB}) async {
    if (_isRunning) return;

    // AUDIT FIX (T-16): previously nothing prevented calling start() with
    // every flag false — status silently said "running" with no
    // subscriptions made and no error surfaced.
    if (!includeAudio && !includeVisionTierA && !includeVisionTierB) {
      _statusController.add('Select at least one modality before starting.');
      return;
    }

    _mergedLog.clear();
    _isRunning = true;
    _statusController.add('Combined test running');

    if (includeAudio) {
      _audioSub = audioController.evidenceStream.listen(_ingest);
      await audioController.start();
    }

    if (includeVisionTierA || includeVisionTierB) {
      visionController.setTierA(includeVisionTierA);
      visionController.setTierB(includeVisionTierB);

      _visionASub = visionController.tierAEvidenceStream.listen((list) {
        for (final e in list) {
          _ingest(e);
        }
      });
      _visionBSub = visionController.tierBEvidenceStream.listen(_ingest);

      await visionController.start();
    }
  }

  void _ingest(Evidence evidence) {
    _mergedLog.add(evidence);
    _mergedLog.sort((a, b) => a.timestamp.compareTo(b.timestamp));
    if (_mergedLog.length > _maxLogEntries) {
      _mergedLog.removeRange(0, _mergedLog.length - _maxLogEntries);
    }
    _mergedFeedController.add(List.unmodifiable(_mergedLog));
    logger.debug('CombinedTestController',
        'Ingested ${evidence.modality.name} evidence at ${evidence.timestamp}');
  }

  Future<void> stop() async {
    _isRunning = false;
    await _audioSub?.cancel();
    await _visionASub?.cancel();
    await _visionBSub?.cancel();
    audioController.stop();
    await visionController.stop();
    _statusController.add('Combined test stopped');
  }

  List<Evidence> get currentLog => List.unmodifiable(_mergedLog);

  Future<void> dispose() async {
    await stop();
    // AUDIT FIX (T-17): previously the owned sub-controllers' internal
    // StreamControllers were never closed — a resource leak. This is now
    // safe post-T-2, since VisionTestController.dispose() no longer
    // disposes the shared EfficientDet/MoViNet model instances.
    await audioController.dispose();
    await visionController.dispose();
    await _mergedFeedController.close();
    await _statusController.close();
  }
}