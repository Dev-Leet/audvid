import 'dart:async';
import '../core/models/evidence.dart';
import '../core/models/test_config.dart';
import '../core/utils/logger.dart';
import '../services/audio/mic_capture_service.dart';
import '../services/audio/audio_preprocessing.dart';
import '../services/audio/yamnet_model.dart';
import '../services/audio/audio_evidence_processor.dart';
import 'perf_stats.dart';

/// Owns the audio test loop: capture -> preprocess -> YAMNet inference ->
/// evidence mapping, repeated on a timer while "running". Exposes raw
/// class scores AND the final Evidence JSON separately so you can see
/// both "what did the model say" and "what would the Risk Engine receive".
class AudioTestController {
  final YamnetModel model;
  final AudioEvidenceProcessor processor;
  final MicCaptureService micCapture;
  final TesterConfig config;
  final PerfStats perfStats = PerfStats();

  final _rawResultsController = StreamController<List<AudioClassResult>>.broadcast();
  final _evidenceController = StreamController<Evidence>.broadcast();
  final _levelController = StreamController<double>.broadcast(); // 0.0-1.0 mic level
  final _statusController = StreamController<String>.broadcast();

  Stream<List<AudioClassResult>> get rawResultsStream => _rawResultsController.stream;
  Stream<Evidence> get evidenceStream => _evidenceController.stream;
  Stream<double> get levelStream => _levelController.stream;
  Stream<String> get statusStream => _statusController.stream;

  bool _isRunning = false;
  bool get isRunning => _isRunning;
  Timer? _loopTimer;

  AudioTestController({
    required this.model,
    required this.processor,
    required this.micCapture,
    this.config = TesterConfig.defaults,
  });

  Future<void> start() async {
    if (_isRunning) return;
    if (!model.isLoaded) {
      _statusController.add('YAMNet model not loaded — cannot start.');
      return;
    }
    final hasPermission = await micCapture.hasPermission();
    if (!hasPermission) {
      _statusController.add('Microphone permission not granted.');
      return;
    }

    _isRunning = true;
    _statusController.add('Running');
    _runLoop(); // fire-and-forget self-rescheduling loop
  }

  void stop() {
    _isRunning = false;
    _loopTimer?.cancel();
    _statusController.add('Stopped');
  }

  Future<void> _runLoop() async {
    while (_isRunning) {
      final cycleStart = DateTime.now();
      try {
        final capture = await micCapture
            .captureWindow(config.audioWindowSeconds)
            .timeout(Duration(seconds: config.audioWindowSeconds + 3));

        if (!_isRunning) break; // stopped mid-capture

        if (capture.pcmBytes.isEmpty) {
          _statusController.add('Empty capture — check mic permission/hardware.');
          perfStats.recordFailure();
          continue;
        }

        // Simple RMS level for the meter widget, computed on raw PCM16
        // BEFORE it's discarded — gives visual confirmation the mic is
        // actually picking up sound, independent of model output.
        _levelController.add(_estimateLevel(capture.pcmBytes));

        final waveform = AudioPreprocessing.prepareForModel(
          pcmBytes: capture.pcmBytes,
          sourceSampleRateHz: capture.sampleRateHz,
          sourceChannels: capture.channels,
          maxWindowSeconds: config.audioWindowSeconds,
        );

        final results = model.infer(waveform);
        _rawResultsController.add(results);

        final evidence = processor.process(results, DateTime.now());
        _evidenceController.add(evidence);

        final latencyMs = DateTime.now().difference(cycleStart).inMilliseconds;
        perfStats.recordLatency(latencyMs);
      } catch (e) {
        logger.error('AudioTestController', 'Cycle failed', e);
        perfStats.recordFailure();
        _statusController.add('Cycle error: $e');
      }
      // Brief gap between windows so the UI can visibly show discrete cycles.
      await Future.delayed(const Duration(milliseconds: 300));
    }
  }

  double _estimateLevel(pcmBytes) {
    if (pcmBytes.isEmpty) return 0.0;
    double sumSquares = 0;
    final sampleCount = pcmBytes.length ~/ 2;
    for (var i = 0; i < sampleCount; i++) {
      final sample = (pcmBytes[i * 2 + 1] << 8) | pcmBytes[i * 2];
      final signed = sample > 32767 ? sample - 65536 : sample;
      sumSquares += (signed / 32768.0) * (signed / 32768.0);
    }
    final rms = sampleCount == 0 ? 0.0 : (sumSquares / sampleCount);
    return rms.clamp(0.0, 1.0) * 4; // scaled for visible meter movement
  }

  Future<void> dispose() async {
    stop();
    await micCapture.dispose();
    await _rawResultsController.close();
    await _evidenceController.close();
    await _levelController.close();
    await _statusController.close();
  }
}