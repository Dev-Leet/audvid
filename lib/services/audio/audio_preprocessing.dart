import 'dart:typed_data';

/// Ported unchanged from the audited main NIRPADAM app.
/// YAMNet's documented input contract: 16 kHz mono waveform, float32
/// samples in [-1.0, 1.0].
class AudioPreprocessing {
  static const int requiredSampleRateHz = 16000;

  static Float32List pcm16ToFloat32(Uint8List pcm16Bytes) {
    final byteData = ByteData.sublistView(pcm16Bytes);
    final sampleCount = pcm16Bytes.length ~/ 2;
    final out = Float32List(sampleCount);
    for (var i = 0; i < sampleCount; i++) {
      out[i] = byteData.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return out;
  }

  static Float32List downmixToMono(Float32List interleaved, int channels) {
    if (channels <= 1) return interleaved;
    final frameCount = interleaved.length ~/ channels;
    final out = Float32List(frameCount);
    for (var i = 0; i < frameCount; i++) {
      double sum = 0;
      for (var c = 0; c < channels; c++) {
        sum += interleaved[i * channels + c];
      }
      out[i] = sum / channels;
    }
    return out;
  }

  static Float32List resampleLinear(Float32List input, int fromHz, int toHz) {
    if (fromHz == toHz || input.isEmpty) return input;
    final ratio = fromHz / toHz;
    final outLength = (input.length / ratio).floor();
    final out = Float32List(outLength);
    for (var i = 0; i < outLength; i++) {
      final srcPos = i * ratio;
      final srcIndex = srcPos.floor();
      final frac = srcPos - srcIndex;
      final a = input[srcIndex];
      final b = (srcIndex + 1 < input.length) ? input[srcIndex + 1] : a;
      out[i] = a + (b - a) * frac;
    }
    return out;
  }

  static Float32List truncateToWindow(Float32List waveform, int maxSeconds) {
    final maxSamples = requiredSampleRateHz * maxSeconds;
    if (waveform.length <= maxSamples) return waveform;
    return Float32List.sublistView(waveform, 0, maxSamples);
  }

  static Float32List prepareForModel({
    required Uint8List pcmBytes,
    required int sourceSampleRateHz,
    required int sourceChannels,
    required int maxWindowSeconds,
  }) {
    var waveform = pcm16ToFloat32(pcmBytes);
    waveform = downmixToMono(waveform, sourceChannels);
    waveform = resampleLinear(waveform, sourceSampleRateHz, requiredSampleRateHz);
    return truncateToWindow(waveform, maxWindowSeconds);
  }
}