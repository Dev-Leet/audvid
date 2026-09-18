import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../../core/utils/logger.dart';

class ActionClassResult {
  final String label;
  final double score;
  const ActionClassResult(this.label, this.score);
}

class MovinetStreamModel {
  static const String assetPath = 'assets/models/movinet_a2_stream_int8.tflite';
  static const String labelMapAssetPath = 'assets/config/kinetics600_labels.txt';
  static const int inputFrameSize = 172;

  final bool outputIsRawLogits;
  final int imageInputTensorIndex;

  /// Confirmed via TF Hub's official MoViNet-A2-Stream/Kinetics-600
  /// classification tutorial: the classifier_head output is a plain
  /// 600-length tensor with NO background/offset class (unlike COCO
  /// detection models, which reserve index 0). The tutorial's own
  /// jumping-jacks reference test indexes directly into the same ordered
  /// label file used here with zero offset and gets the correct answer.
  /// Kept as a parameter — not hardcoded — because that citation covers
  /// the source SavedModel, not necessarily whatever specific .tflite
  /// conversion you actually downloaded; run the calibration test below
  /// before trusting it for your file.
  final int labelIndexOffset;

  Interpreter? _interpreter;
  Map<int, dynamic> _states = {};
  List<int> _stateInputIndices = [];
  List<String> _labels = [];
  bool _tensorMappingVerifiedSafe = false;

  MovinetStreamModel({
    this.outputIsRawLogits = true,
    this.imageInputTensorIndex = 0,
    this.labelIndexOffset = 0,
  });

  bool get isLoaded => _interpreter != null && _tensorMappingVerifiedSafe;
  List<String> get labels => _labels;

  Future<void> load() async {
    try {
      _interpreter = await Interpreter.fromAsset(assetPath);
      _labels = await _loadLabels();

      final inputCount = _interpreter!.getInputTensors().length;
      final outputCount = _interpreter!.getOutputTensors().length;

      logger.info('MovinetStreamModel',
          'Loaded with $inputCount inputs / $outputCount outputs. VERIFY against model card.');
      for (var i = 0; i < inputCount; i++) {
        final t = _interpreter!.getInputTensor(i);
        logger.debug('MovinetStreamModel', 'Input[$i] name=${t.name} shape=${t.shape}');
      }
      for (var i = 0; i < outputCount; i++) {
        final t = _interpreter!.getOutputTensor(i);
        logger.debug('MovinetStreamModel', 'Output[$i] name=${t.name} shape=${t.shape}');
      }

      if (inputCount != outputCount) {
        logger.error('MovinetStreamModel',
            'Input tensor count ($inputCount) != output tensor count '
            '($outputCount) — the assumed state-mirroring pattern does '
            'NOT hold for this model file. Refusing to run to avoid '
            'silently corrupting recurrent state. Provide a verified '
            'explicit mapping before using this model.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      // NEW: verify the classifier output tensor's DTYPE and class count
      // before trusting it. The asset is named "...int8.tflite" — if this
      // model's output tensor is genuinely quantized (TensorType.int8 /
      // uint8) rather than float32, every value in processFrame()'s
      // List<double> output buffer would be the wrong buffer type
      // entirely, independent of label ordering. Refuse to run rather
      // than silently misinterpreting raw quantized integers as
      // probabilities/logits.
      final classifierTensor = _interpreter!.getOutputTensor(imageInputTensorIndex);
      logger.info('MovinetStreamModel',
          'Classifier output tensor: type=${classifierTensor.type}, '
          'shape=${classifierTensor.shape}');

      final isFloatOutput = classifierTensor.type.toString().toLowerCase().contains('float');
      if (!isFloatOutput) {
        logger.error('MovinetStreamModel',
            'Classifier output tensor type is ${classifierTensor.type}, not '
            'float32. This model\'s output is genuinely quantized — the '
            'current float-based softmax/probability pipeline will '
            'misinterpret raw integer values. Refusing to run until '
            'explicit dequantization (scale=${classifierTensor.params.scale}, '
            'zeroPoint=${classifierTensor.params.zeroPoint}) is implemented '
            'and verified.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      final outputClassCount = classifierTensor.shape.last;
      final expectedCount = _labels.length + labelIndexOffset;
      if (outputClassCount != expectedCount) {
        logger.error('MovinetStreamModel',
            'Classifier output has $outputClassCount classes but the label '
            'file + offset implies $expectedCount ($_labels.length labels, '
            'offset=$labelIndexOffset). These MUST match exactly or every '
            'prediction will be mislabeled. Refusing to run.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      logger.info('MovinetStreamModel',
          'Verified: float32 output, $outputClassCount classes matches '
          '$_labels.length labels + offset=$labelIndexOffset. Safe to run — '
          'still recommend the jumping-jacks calibration test once before '
          'trusting field results.');

      _stateInputIndices = List.generate(inputCount, (i) => i)
          .where((i) => i != imageInputTensorIndex)
          .toList();

      _initializeStreamState();
      _tensorMappingVerifiedSafe = true;
    } catch (e) {
      logger.error('MovinetStreamModel', 'load() failed', e);
      _interpreter = null;
      _tensorMappingVerifiedSafe = false;
    }
  }

  Future<List<String>> _loadLabels() async {
    try {
      final raw = await rootBundle.loadString(labelMapAssetPath);
      final lines = raw.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
      if (lines.isEmpty) throw Exception('Empty label file');
      return lines;
    } catch (e) {
      logger.warning('MovinetStreamModel', 'Could not load Kinetics-600 labels — using placeholder.', e);
      return const ['unknown_action'];
    }
  }

  void _initializeStreamState() {
    final interpreter = _interpreter;
    if (interpreter == null) return;
    _states = {};
    for (final i in _stateInputIndices) {
      final tensor = interpreter.getInputTensor(i);
      _states[i] = List.filled(tensor.shape.reduce((a, b) => a * b), 0).reshape(tensor.shape);
    }
  }

  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(max);
    final exps = logits.map((l) => exp(l - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  List<ActionClassResult> processFrame(Uint8List rgbFrame) {
    final interpreter = _interpreter;
    if (interpreter == null || !_tensorMappingVerifiedSafe) return const [];

    try {
      final imageInput = rgbFrame.reshape([1, 1, inputFrameSize, inputFrameSize, 3]);
      final inputs = <int, Object>{imageInputTensorIndex: imageInput};
      inputs.addAll(_states.map((k, v) => MapEntry(k, v)));

      final outputs = <int, Object>{};
      final logitsShape = interpreter.getOutputTensor(imageInputTensorIndex).shape;
      outputs[imageInputTensorIndex] =
          List.filled(logitsShape.reduce((a, b) => a * b), 0.0).reshape(logitsShape);
      for (final stateKey in _states.keys) {
        final shape = interpreter.getOutputTensor(stateKey).shape;
        outputs[stateKey] = List.filled(shape.reduce((a, b) => a * b), 0).reshape(shape);
      }

      interpreter.runForMultipleInputs(inputs.values.toList(), outputs);

      for (final stateKey in _states.keys) {
        _states[stateKey] = outputs[stateKey];
      }

      final rawLogits = (outputs[imageInputTensorIndex] as List).cast<List>().first as List;
      final logitsAsDoubles = rawLogits.map((v) => (v as num).toDouble()).toList();
      final probabilities = outputIsRawLogits ? _softmax(logitsAsDoubles) : logitsAsDoubles;

      // labelIndexOffset applied here — confirmed 0 for the official
      // source model (see class-level doc comment), but kept as an actual
      // arithmetic offset rather than assumed away, so a differently
      // exported .tflite file can be corrected with one constructor
      // argument instead of a code change.
      final results = <ActionClassResult>[];
      for (var i = 0; i < probabilities.length; i++) {
        final labelIdx = i - labelIndexOffset;
        if (labelIdx < 0 || labelIdx >= _labels.length) continue;
        results.add(ActionClassResult(_labels[labelIdx], probabilities[i]));
      }
      results.sort((a, b) => b.score.compareTo(a.score));
      return results.take(5).toList();
    } catch (e) {
      logger.error('MovinetStreamModel', 'processFrame() failed', e);
      return const [];
    }
  }

  void resetStream() {
    if (_tensorMappingVerifiedSafe) _initializeStreamState();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _states.clear();
    _tensorMappingVerifiedSafe = false;
  }
}