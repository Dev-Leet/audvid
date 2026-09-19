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
  
  static const int inputFrameSize = 224;

  final bool outputIsRawLogits;
  final int labelIndexOffset;

  Interpreter? _interpreter;
  Map<int, dynamic> _states = {};
  
  static const int _imageInputIndex = 60;
  static const int _classifierOutputIndex = 13;

  /// Exact mapping of Dart input index -> Dart output index for the recurrent state tensors.
  /// Extracted via Python directly from the .tflite flatbuffer's SignatureDef.
  /// DO NOT "auto-discover" or zip these arrays! TFLite scrambles the underlying tensor indices,
  /// and many blocks share identical buffer shapes, making shape-based pairing impossible.
  static const Map<int, int> _stateInToOut = {
    25: 26, // state_block0_layer0_pool_buffer
    27: 28, // state_block0_layer0_pool_frame_count
    26: 27, // state_block0_layer1_pool_buffer
    31: 32, // state_block0_layer1_pool_frame_count
    52: 53, // state_block0_layer1_stream_buffer
    19: 20, // state_block0_layer2_pool_buffer
    73: 73, // state_block0_layer2_pool_frame_count
    28: 29, // state_block0_layer2_stream_buffer
    49: 50, // state_block1_layer0_pool_buffer
    30: 31, // state_block1_layer0_pool_frame_count
    43: 44, // state_block1_layer0_stream_buffer
    23: 24, // state_block1_layer1_pool_buffer
    37: 38, // state_block1_layer1_pool_frame_count
    1: 1,   // state_block1_layer1_stream_buffer
    61: 61, // state_block1_layer2_pool_buffer
    59: 60, // state_block1_layer2_pool_frame_count
    2: 2,   // state_block1_layer2_stream_buffer
    64: 64, // state_block1_layer3_pool_buffer
    17: 18, // state_block1_layer3_pool_frame_count
    50: 51, // state_block1_layer3_stream_buffer
    46: 47, // state_block1_layer4_pool_buffer
    38: 39, // state_block1_layer4_pool_frame_count
    39: 40, // state_block1_layer4_stream_buffer
    62: 62, // state_block2_layer0_pool_buffer
    34: 35, // state_block2_layer0_pool_frame_count
    53: 54, // state_block2_layer0_stream_buffer
    13: 14, // state_block2_layer1_pool_buffer
    72: 72, // state_block2_layer1_pool_frame_count
    14: 15, // state_block2_layer1_stream_buffer
    56: 57, // state_block2_layer2_pool_buffer
    33: 34, // state_block2_layer2_pool_frame_count
    11: 11, // state_block2_layer2_stream_buffer
    40: 41, // state_block2_layer3_pool_buffer
    69: 69, // state_block2_layer3_pool_frame_count
    18: 19, // state_block2_layer3_stream_buffer
    29: 30, // state_block2_layer4_pool_buffer
    66: 66, // state_block2_layer4_pool_frame_count
    5: 5,   // state_block2_layer4_stream_buffer
    32: 33, // state_block3_layer0_pool_buffer
    51: 52, // state_block3_layer0_pool_frame_count
    10: 10, // state_block3_layer0_stream_buffer
    44: 45, // state_block3_layer1_pool_buffer
    3: 3,   // state_block3_layer1_pool_frame_count
    15: 16, // state_block3_layer1_stream_buffer
    4: 4,   // state_block3_layer2_pool_buffer
    24: 25, // state_block3_layer2_pool_frame_count
    41: 42, // state_block3_layer2_stream_buffer
    71: 71, // state_block3_layer3_pool_buffer
    6: 6,   // state_block3_layer3_pool_frame_count
    22: 23, // state_block3_layer3_stream_buffer
    65: 65, // state_block3_layer4_pool_buffer
    45: 46, // state_block3_layer4_pool_frame_count
    70: 70, // state_block3_layer5_pool_buffer
    68: 68, // state_block3_layer5_pool_frame_count
    54: 55, // state_block3_layer5_stream_buffer
    7: 7,   // state_block4_layer0_pool_buffer
    16: 17, // state_block4_layer0_pool_frame_count
    42: 43, // state_block4_layer0_stream_buffer
    58: 59, // state_block4_layer1_pool_buffer
    0: 0,   // state_block4_layer1_pool_frame_count
    67: 67, // state_block4_layer2_pool_buffer
    36: 37, // state_block4_layer2_pool_frame_count
    48: 49, // state_block4_layer3_pool_buffer
    63: 63, // state_block4_layer3_pool_frame_count
    9: 9,   // state_block4_layer4_pool_buffer
    21: 22, // state_block4_layer4_pool_frame_count
    35: 36, // state_block4_layer5_pool_buffer
    20: 21, // state_block4_layer5_pool_frame_count
    12: 12, // state_block4_layer5_stream_buffer
    55: 56, // state_block4_layer6_pool_buffer
    47: 48, // state_block4_layer6_pool_frame_count
    57: 58, // state_head_pool_buffer
    8: 8,   // state_head_pool_frame_count
  };

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
          'Loaded with $inputCount inputs / $outputCount outputs.');

      if (inputCount != 74 || outputCount != 74) {
        logger.error('MovinetStreamModel',
            'Input/output count mismatch vs expected 74. The hardcoded state tensor mapping '
            'is strictly valid ONLY for the specific movinet_a2_stream_int8.tflite export. '
            'Refusing to run.');
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
            'float32. Refusing to run until explicit dequantization is implemented.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      final outputClassCount = classifierTensor.shape.last;
      final expectedCount = _labels.length + labelIndexOffset;
      if (outputClassCount != expectedCount) {
        logger.error('MovinetStreamModel',
            'Classifier output has $outputClassCount classes but the label '
            'file + offset implies $expectedCount. Refusing to run.');
        _tensorMappingVerifiedSafe = false;
        return;
      }

      logger.info('MovinetStreamModel', 'Verified: float32 output, $outputClassCount classes. Safe to run.');

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
    for (final inIdx in _stateInToOut.keys) {
      final tensor = interpreter.getInputTensor(inIdx);
      _states[inIdx] = List.filled(tensor.shape.reduce((a, b) => a * b), 0).reshape(tensor.shape);
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
      final inputs = List<Object>.filled(74, []);
      inputs[_imageInputIndex] = imageInput;
      for (final inIdx in _stateInToOut.keys) {
        inputs[inIdx] = _states[inIdx];
      }

      final outputs = <int, Object>{};
      final logitsShape = interpreter.getOutputTensor(_classifierOutputIndex).shape;
      outputs[_classifierOutputIndex] =
          List.filled(logitsShape.reduce((a, b) => a * b), 0.0).reshape(logitsShape);
          
      for (final outIdx in _stateInToOut.values) {
        final shape = interpreter.getOutputTensor(outIdx).shape;
        outputs[outIdx] = List.filled(shape.reduce((a, b) => a * b), 0).reshape(shape);
      }

      interpreter.runForMultipleInputs(inputs, outputs);

      // Update state for the next frame
      for (final entry in _stateInToOut.entries) {
        _states[entry.key] = outputs[entry.value];
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