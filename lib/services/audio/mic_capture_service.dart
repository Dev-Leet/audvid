import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart' show getTemporaryDirectory;
import '../../core/utils/logger.dart';

class MicCaptureResult {
  final Uint8List pcmBytes;
  final int sampleRateHz;
  final int channels;
  const MicCaptureResult(this.pcmBytes, this.sampleRateHz, this.channels);
}

class MicCaptureService {
  final AudioRecorder _recorder = AudioRecorder();
  static const int _requestedSampleRate = 16000;
  static const int _requestedChannels = 1;

  /// AUDIT FIX (T-19): previously checked via `record`'s own
  /// `AudioRecorder.hasPermission()`, while `main.dart` separately checked
  /// via `permission_handler` — two different libraries' views of the same
  /// OS permission. `permission_handler` is now the single source of truth
  /// everywhere in this app.
  Future<bool> hasPermission() => Permission.microphone.isGranted;

  Future<MicCaptureResult> captureWindow(int seconds) async {
    if (!await hasPermission()) {
      logger.warning('MicCaptureService', 'Microphone permission not granted');
      return MicCaptureResult(Uint8List(0), _requestedSampleRate, _requestedChannels);
    }

    final tempDir = await getTemporaryDirectory();
    final path =
        '${tempDir.path}/nirpadam_tester_${DateTime.now().millisecondsSinceEpoch}.wav';

    try {
      // AUDIT FIX (T-10): start() previously had no timeout — a stalled
      // platform channel would hang captureWindow() indefinitely.
      await _recorder
          .start(
            const RecordConfig(
              encoder: AudioEncoder.wav,
              sampleRate: _requestedSampleRate,
              numChannels: _requestedChannels,
            ),
            path: path,
          )
          .timeout(const Duration(seconds: 3),
              onTimeout: () => throw TimeoutException('recorder.start() timed out'));

      // AUDIT FIX (T-10): the original wrapped this delay in
      // `.timeout(Duration(seconds: seconds + 2))` — a timeout LONGER than
      // the delay it wraps can never fire and provided no real protection.
      // The actual protection is the outer `.timeout()` AudioTestController
      // already applies to the whole `captureWindow()` call.
      await Future.delayed(Duration(seconds: seconds));

      final resultPath = await _recorder
          .stop()
          .timeout(const Duration(seconds: 3),
              onTimeout: () => throw TimeoutException('recorder.stop() timed out'));

      if (resultPath == null) {
        logger.warning('MicCaptureService', 'Recorder returned no file path');
        return MicCaptureResult(Uint8List(0), _requestedSampleRate, _requestedChannels);
      }

      final file = File(resultPath);
      if (!await file.exists()) {
        return MicCaptureResult(Uint8List(0), _requestedSampleRate, _requestedChannels);
      }

      final bytes = await file.readAsBytes();
      final pcm = _extractPcmFromWav(bytes);
      await file.delete().catchError((_) => file); // don't retain raw audio

      return MicCaptureResult(pcm, _requestedSampleRate, _requestedChannels);
    } catch (e) {
      logger.error('MicCaptureService', 'captureWindow failed', e);
      try {
        await _recorder.stop().timeout(const Duration(seconds: 2));
      } catch (_) {}
      return MicCaptureResult(Uint8List(0), _requestedSampleRate, _requestedChannels);
    }
  }

  /// AUDIT FIX (T-10): the previous implementation assumed a fixed 44-byte
  /// canonical header. Real WAV files can carry extra chunks (e.g. LIST,
  /// fmt extensions) before `data`, which would silently shift/corrupt the
  /// PCM samples with no error. This scans the RIFF chunk structure for
  /// the actual `data` subchunk instead of assuming its offset.
  Uint8List _extractPcmFromWav(Uint8List wavBytes) {
    if (wavBytes.length < 12) return Uint8List(0);

    final riffId = String.fromCharCodes(wavBytes.sublist(0, 4));
    final waveId = String.fromCharCodes(wavBytes.sublist(8, 12));
    if (riffId != 'RIFF' || waveId != 'WAVE') {
      logger.warning('MicCaptureService', 'Not a valid RIFF/WAVE file');
      return Uint8List(0);
    }

    var offset = 12;
    while (offset + 8 <= wavBytes.length) {
      final chunkId = String.fromCharCodes(wavBytes.sublist(offset, offset + 4));
      final chunkSize =
          ByteData.sublistView(wavBytes, offset + 4, offset + 8).getUint32(0, Endian.little);
      final dataStart = offset + 8;

      if (chunkId == 'data') {
        final dataEnd = (dataStart + chunkSize).clamp(0, wavBytes.length);
        return Uint8List.sublistView(wavBytes, dataStart, dataEnd);
      }

      // Chunks are word-aligned (padded to an even byte count).
      offset = dataStart + chunkSize + (chunkSize % 2);
    }

    logger.warning('MicCaptureService', 'No "data" chunk found in WAV file');
    return Uint8List(0);
  }

  Future<void> dispose() async {
    try {
      if (await _recorder.isRecording()) {
        await _recorder.stop().timeout(const Duration(seconds: 2));
      }
    } catch (_) {}
    _recorder.dispose();
  }
}