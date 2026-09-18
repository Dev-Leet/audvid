import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../../core/utils/logger.dart';

class Detection {
  final String label;
  final double confidence;
  final List<double> boundingBox; // [ymin, xmin, ymax, xmax], normalized 0-1
  const Detection(this.label, this.confidence, this.boundingBox);
}

class EfficientDetModel {
  static const String assetPath = 'assets/models/efficientdet_lite0.tflite';
  static const String labelMapAssetPath = 'assets/config/coco_labelmap.txt';
  static const double minConfidence = 0.5;

  Interpreter? _interpreter;
  List<String> _labels = [];
  int _maxDetections = 25;
  final int labelIndexOffset;

  EfficientDetModel({this.labelIndexOffset = 1});

  bool get isLoaded => _interpreter != null;

  Future<void> load() async {
    try {
      _interpreter = await Interpreter.fromAsset(assetPath);
      _labels = await _loadLabelMap();
      final boxesShape = _interpreter!.getOutputTensor(0).shape;
      if (boxesShape.length >= 2) _maxDetections = boxesShape[1];
      logger.info('EfficientDetModel', 'Loaded, max detections=$_maxDetections');
    } catch (e) {
      logger.error('EfficientDetModel', 'load() failed', e);
      _interpreter = null;
    }
  }

  Future<List<String>> _loadLabelMap() async {
    try {
      final raw = await rootBundle.loadString(labelMapAssetPath);
      final lines = raw.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
      if (lines.isEmpty) throw Exception('Label map asset is empty');
      return lines;
    } catch (e) {
      logger.warning('EfficientDetModel', 'Could not load label map — using minimal fallback.', e);
      return const ['???', 'person'];
    }
  }

  String _labelFor(int rawClassIdx) {
    final adjusted = rawClassIdx + labelIndexOffset;
    if (adjusted >= 0 && adjusted < _labels.length) return _labels[adjusted];
    return 'unknown_id_$rawClassIdx';
  }

  List<Detection> infer(Uint8List rgb320x320) {
    final interpreter = _interpreter;
    if (interpreter == null) return const [];

    try {
      final input = rgb320x320.reshape([1, 320, 320, 3]);
      final outputBoxes =
          List.generate(1, (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)));
      final outputClasses = List.generate(1, (_) => List.filled(_maxDetections, 0.0));
      final outputScores = List.generate(1, (_) => List.filled(_maxDetections, 0.0));
      final outputCount = List.filled(1, 0.0);

      interpreter.runForMultipleInputs(
          [input], {0: outputBoxes, 1: outputClasses, 2: outputScores, 3: outputCount});

      final results = <Detection>[];
      final count = outputCount[0].toInt().clamp(0, _maxDetections);
      for (var i = 0; i < count; i++) {
        final score = outputScores[0][i];
        if (score < minConfidence) continue;
        results.add(Detection(_labelFor(outputClasses[0][i].toInt()), score, outputBoxes[0][i]));
      }
      return results;
    } catch (e) {
      logger.error('EfficientDetModel', 'infer() failed', e);
      return const [];
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}