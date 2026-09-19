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
  
  // The input frame size for this specific MoViNet export. 
  // Adjusted from 172 to 224 based on the actual tensor shape logs.
  static const int inputFrameSize = 224;

  final bool outputIsRawLogits;
  final int labelIndexOffset;

  Interpreter? _interpreter;
  Map<int, dynamic> _states = {};
  
  int _imageInputIndex = -1;
  int _classifierOutputIndex = -1;
  List<int> _stateInputIndices = [];
  List<int> _stateOutputIndices = [];
  
  List<String> _labels = [];
  bool _tensorMappingVerifiedSafe = false;

  MovinetStreamModel({
    this.outputIsRawLogits = true,
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
          'Loaded with $inputCount inputs / $outputCount outputs. Discovering indices...');
          
      // Auto-discover the image input and state inputs
      _stateInputIndices = [];
      for (var i = 0; i < inputCount; i++) {
        final t = _interpreter!.getInputTensor(i);
        // Look for the image tensor by shape: [1, 1, 224, 224, 3] or [1, 1, 172, 172, 3]
        if (t.shape.length == 5 && t.shape.last == 3) {
          _imageInputIndex = i;
          logger.info('MovinetStreamModel', 'Found image input at index $i (shape: ${t.shape})');
        } else {
          _stateInputIndices.add(i);
        }
      }

      // Auto-discover the classifier output and state outputs
      _stateOutputIndices = [];
      for (var i = 0; i < outputCount; i++) {
        final t = _interpreter!.getOutputTensor(i);
        // Look for the classifier output by shape: [1, 600]
        if (t.shape.length == 2 && t.shape.last >= 400 && t.shape.last <= 700) {
          _classifierOutputIndex = i;
          logger.info('MovinetStreamModel', 'Found classifier output at index $i (shape: ${t.shape})');
        } else {
          _stateOutputIndices.add(i);
        }
      }

      if (_imageInputIndex == -1 || _classifierOutputIndex == -1) {
        logger.error('MovinetStreamModel', 'Could not auto-discover image input or classifier output tensors. Refusing to run.');
        _tensorMappingVerifiedSafe = false;
        return;
      }
      
      if (_stateInputIndices.length != _stateOutputIndices.length) {
        logger.error('MovinetStreamModel',
            'State input count (${_stateInputIndices.length}) != state output count '
            '(${_stateOutputIndices.length}). Refusing to run to avoid state corruption.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      final classifierTensor = _interpreter!.getOutputTensor(_classifierOutputIndex);
      logger.info('MovinetStreamModel',
          'Classifier output tensor: type=${classifierTensor.type}, '
          'shape=${classifierTensor.shape}, '
          'scale=${classifierTensor.params.scale}, '
          'zeroPoint=${classifierTensor.params.zeroPoint}');

      final isFloatOutput = classifierTensor.type.toString().toLowerCase().contains('float');
      if (!isFloatOutput) {
        logger.error('MovinetStreamModel',
            'Classifier output tensor type is ${classifierTensor.type}, not '
            'float32. This model\'s output is genuinely quantized. Refusing to run until '
            'explicit dequantization (scale=${classifierTensor.params.scale}, '
            'zeroPoint=${classifierTensor.params.zeroPoint}) is implemented.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      final outputClassCount = classifierTensor.shape.last;
      final expectedCount = _labels.length + labelIndexOffset;
      if (outputClassCount != expectedCount) {
        logger.error('MovinetStreamModel',
            'Classifier output has $outputClassCount classes but the label '
            'file + offset implies $expectedCount ($_labels.length labels, '
            'offset=$labelIndexOffset). These MUST match exactly. Refusing to run.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      logger.info('MovinetStreamModel', 'Verified: float32 output, $outputClassCount classes matches labels. Safe to run.');

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
      
      // Assemble inputs in the exact order of interpreter.getInputTensors()
      final inputs = List<Object>.filled(_interpreter!.getInputTensors().length, []);
      inputs[_imageInputIndex] = imageInput;
      for (final inIdx in _stateInputIndices) {
        inputs[inIdx] = _states[inIdx];
      }

      final outputs = <int, Object>{};
      final logitsShape = interpreter.getOutputTensor(_classifierOutputIndex).shape;
      outputs[_classifierOutputIndex] =
          List.filled(logitsShape.reduce((a, b) => a * b), 0.0).reshape(logitsShape);
          
      for (final outIdx in _stateOutputIndices) {
        final shape = interpreter.getOutputTensor(outIdx).shape;
        outputs[outIdx] = List.filled(shape.reduce((a, b) => a * b), 0).reshape(shape);
      }

      interpreter.runForMultipleInputs(inputs, outputs);

      // Update state for the next frame using the parallel arrays
      for (var i = 0; i < _stateInputIndices.length; i++) {
        final inIdx = _stateInputIndices[i];
        final outIdx = _stateOutputIndices[i];
        _states[inIdx] = outputs[outIdx];
      }

      final rawLogits = (outputs[_classifierOutputIndex] as List).cast<List>().first;
      final logitsAsDoubles = rawLogits.map((v) => (v as num).toDouble()).toList();
      final probabilities = outputIsRawLogits ? _softmax(logitsAsDoubles) : logitsAsDoubles;

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