import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../../core/utils/logger.dart';

class AudioClassResult {
  final String label;
  final double score;
  const AudioClassResult(this.label, this.score);
}

class YamnetModel {
  static const String assetPath = 'assets/models/yamnet.tflite';
  static const String labelMapAssetPath = 'assets/config/yamnet_class_map.csv';

  Interpreter? _interpreter;
  List<String> _labels = [];
  int _outputTensorCount = 1; // AUDIT FIX (T-11): verified at load, not assumed
  bool get isLoaded => _interpreter != null;
  List<String> get labels => _labels;

  Future<void> load() async {
    try {
      _interpreter = await Interpreter.fromAsset(assetPath);
      _labels = await _loadLabels();

      // AUDIT FIX (T-11): log and record the REAL output tensor count.
      // `infer()` used the single-input/single-output convenience API
      // (`interpreter.run`), which silently assumes exactly one output —
      // unverified against the actual shipped model. Some YAMNet exports
      // include extra outputs (e.g. embeddings/spectrogram) alongside
      // scores.
      _outputTensorCount = _interpreter!.getOutputTensors().length;
      logger.info('YamnetModel',
          'Loaded with ${_labels.length} labels, $_outputTensorCount output tensor(s). '
          'VERIFY against the model card if this is not 1.');
      if (_outputTensorCount != 1) {
        logger.warning('YamnetModel',
            'Model has $_outputTensorCount outputs, not the assumed 1 — '
            'infer() will attempt to locate the scores tensor by shape.');
      }
    } catch (e) {
      logger.error('YamnetModel', 'load() failed', e);
      _interpreter = null;
    }
  }

  /// AUDIT FIX (T-12): the previous fallback returned a small hardcoded
  /// label list mapped POSITIONALLY onto the model's real output indices
  /// — near-guaranteed to be wrong, and displayed with no visual
  /// indication that it's fabricated (e.g. "Screaming: 82%" for whatever
  /// the model's real index 1 actually is). `_labelsAreReal` now tracks
  /// whether the genuine label file loaded, and `resolveLabel()` returns
  /// an explicit, honest placeholder per index when it didn't — never a
  /// specific concept name that could be mistaken for a real result.
  bool _labelsAreReal = false;
  bool get labelsAreReal => _labelsAreReal;

  Future<List<String>> _loadLabels() async {
    try {
      final raw = await rootBundle.loadString(labelMapAssetPath);
      final lines = raw.split('\n').where((l) => l.trim().isNotEmpty).toList();
      final labels = <String>[];
      // Skip header row ("index,mid,display_name").
      for (final line in lines.skip(1)) {
        final fields = _parseCsvLine(line);
        if (fields.length >= 3) {
          // display_name may itself contain commas (already handled by
          // _parseCsvLine), so join anything past index/mid just in case
          // of a stray unescaped comma, then trim.
          labels.add(fields.sublist(2).join(',').trim());
        }
      }
      if (labels.isEmpty) throw Exception('Parsed 0 labels from class map');
      _labelsAreReal = true;
      return labels;
    } catch (e) {
      logger.warning('YamnetModel',
          'Could not load the real yamnet_class_map.csv — results will show '
          'as "unresolved_index_N" rather than fabricated label names.', e);
      _labelsAreReal = false;
      return List.generate(521, (i) => 'unresolved_index_$i');
    }
  }

  /// Wiring the real yamnet_class_map.csv exposed a parsing bug: many
  /// display names contain embedded commas inside quotes (e.g.
  /// `"Child speech, kid speaking"`). A naive `line.split(',')` breaks
  /// those into extra parts, and rejoining with `.join(',')` reconstructs
  /// the comma but leaves the surrounding literal quote characters in the
  /// label text (e.g. `"Child speech, kid speaking"` instead of
  /// `Child speech, kid speaking`). This is a minimal quote-aware CSV
  /// tokenizer — sufficient for this file's shape (no escaped `""` quotes
  /// appear in the real data, but handled defensively anyway).
  List<String> _parseCsvLine(String line) {
    final fields = <String>[];
    final buffer = StringBuffer();
    var inQuotes = false;

    for (var i = 0; i < line.length; i++) {
      final char = line[i];
      if (char == '"') {
        if (inQuotes && i + 1 < line.length && line[i + 1] == '"') {
          buffer.write('"'); // escaped quote
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char == ',' && !inQuotes) {
        fields.add(buffer.toString());
        buffer.clear();
      } else {
        buffer.write(char);
      }
    }
    fields.add(buffer.toString());
    return fields;
  }

  List<AudioClassResult> infer(Float32List waveform) {
    final interpreter = _interpreter;
    if (interpreter == null || waveform.isEmpty) return const [];

    try {
      interpreter.resizeInputTensor(0, [waveform.length]);
      interpreter.allocateTensors();
      final input = waveform.reshape([waveform.length]);

      List<List<double>> output;

      if (_outputTensorCount <= 1) {
        // Original, verified-common case: exactly one output tensor.
        final outputShape = interpreter.getOutputTensor(0).shape;
        output = List.generate(outputShape[0], (_) => List.filled(outputShape[1], 0.0));
        interpreter.run(input, output);
      } else {
        // AUDIT FIX (T-11): multi-output export — run all outputs via
        // runForMultipleInputs, then pick the tensor whose last dimension
        // looks like a class-scores vector (AudioSet has 521 classes) so
        // an embeddings/spectrogram output isn't mistaken for scores.
        final outputs = <int, Object>{};
        int? scoresTensorIndex;
        for (var i = 0; i < _outputTensorCount; i++) {
          final shape = interpreter.getOutputTensor(i).shape;
          outputs[i] = List.generate(shape[0], (_) => List.filled(shape.last, 0.0));
          if (shape.last >= 400 && shape.last <= 700) {
            scoresTensorIndex = i; // plausible AudioSet-scores-shaped tensor
          }
        }
        interpreter.runForMultipleInputs([input], outputs);

        if (scoresTensorIndex == null) {
          logger.error('YamnetModel',
              'Could not identify a scores tensor among $_outputTensorCount outputs.');
          return const [];
        }
        output = (outputs[scoresTensorIndex] as List).cast<List<double>>();
      }

      final numClasses = output.isEmpty ? 0 : output.first.length;
      final pooled = List.filled(numClasses, 0.0);
      for (final frame in output) {
        for (var c = 0; c < numClasses; c++) {
          pooled[c] += frame[c] / output.length;
        }
      }

      final results = <AudioClassResult>[];
      for (var c = 0; c < numClasses && c < _labels.length; c++) {
        results.add(AudioClassResult(_labels[c], pooled[c]));
      }
      results.sort((a, b) => b.score.compareTo(a.score));
      return results.take(30).toList();
    } catch (e) {
      logger.error('YamnetModel', 'infer() failed', e);
      return const [];
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}