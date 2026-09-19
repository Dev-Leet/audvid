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
      
      // The specific EfficientDet-Lite0 model downloaded only returns 2 outputs:
      // Output 0: classes/scores (shape [1, 19206, 90])
      // Output 1: bounding boxes (shape [1, 19206, 4])
      // It does NOT do NMS (Non-Maximum Suppression) internally. We must process the raw tensors.
      
      final outputClassesScores = List.generate(1, (_) => List.generate(_maxDetections, (_) => List.filled(90, 0.0)));
      final outputBoxes = List.generate(1, (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)));

      interpreter.runForMultipleInputs(
          [input], {0: outputClassesScores, 1: outputBoxes});

      final results = <Detection>[];
      
      // Simple manual NMS and score filtering
      for (var i = 0; i < _maxDetections; i++) {
        double maxScore = 0;
        int maxClassId = -1;
        
        for (var c = 0; c < 90; c++) {
          final score = outputClassesScores[0][i][c];
          if (score > maxScore) {
            maxScore = score;
            maxClassId = c;
          }
        }
        
        if (maxScore >= minConfidence) {
          final box = outputBoxes[0][i];
          // TFLite bounding boxes for EfficientDet are typically output as
          // [y_min, x_min, y_max, x_max] but depending on the normalization,
          // they might be center_y, center_x, height, width, OR they might
          // NOT be normalized to 0-1 and instead mapped to the 320x320 grid.
          // Let's ensure they are normalized properly.
          
          double yMin = box[0];
          double xMin = box[1];
          double yMax = box[2];
          double xMax = box[3];
          
          // If the box values are larger than 1.0, they are absolute pixel values (0-320).
          // We must normalize them back to 0.0 - 1.0 for the UI to draw correctly over any screen size.
          if (yMax > 2.0 || xMax > 2.0) {
            yMin = yMin / 320.0;
            xMin = xMin / 320.0;
            yMax = yMax / 320.0;
            xMax = xMax / 320.0;
          }

          // Filter out completely invalid boxes
          if (yMin < yMax && xMin < xMax) {
             results.add(Detection(_labelFor(maxClassId), maxScore, [yMin, xMin, yMax, xMax]));
          }
        }
      }
      
      // NMS (IoU thresholding) to remove duplicates
      results.sort((a, b) => b.confidence.compareTo(a.confidence));
      final nmsResults = <Detection>[];
      for (final current in results) {
        bool drop = false;
        for (final kept in nmsResults) {
          if (current.label == kept.label && _iou(current.boundingBox, kept.boundingBox) > 0.5) {
            drop = true;
            break;
          }
        }
        if (!drop) nmsResults.add(current);
      }
      
      return nmsResults;
    } catch (e) {
      logger.error('EfficientDetModel', 'infer() failed', e);
      return const [];
    }
  }

  double _iou(List<double> a, List<double> b) {
    final yMin = a[0] > b[0] ? a[0] : b[0];
    final xMin = a[1] > b[1] ? a[1] : b[1];
    final yMax = a[2] < b[2] ? a[2] : b[2];
    final xMax = a[3] < b[3] ? a[3] : b[3];
    final interH = (yMax - yMin).clamp(0, double.infinity);
    final interW = (xMax - xMin).clamp(0, double.infinity);
    final interArea = interH * interW;
    final areaA = (a[2] - a[0]) * (a[3] - a[1]);
    final areaB = (b[2] - b[0]) * (b[3] - b[1]);
    final union = areaA + areaB - interArea;
    return union <= 0 ? 0 : interArea / union;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}