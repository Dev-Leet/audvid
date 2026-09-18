import 'package:flutter/material.dart';
import '../services/vision/efficientdet_model.dart';
import '../services/vision/object_tracker.dart';

/// Paints EfficientDet-Lite0 bounding boxes and ObjectTracker track IDs
/// over the live camera preview so you can visually confirm both
/// detection AND tracking are behaving correctly before trusting the
/// heuristic evidence derived from them.
class DetectionOverlayPainter extends CustomPainter {
  final List<Detection> detections;
  final List<TrackedObject> tracks;

  DetectionOverlayPainter({required this.detections, required this.tracks});

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.boundingBox[1] * size.width,
        d.boundingBox[0] * size.height,
        d.boundingBox[3] * size.width,
        d.boundingBox[2] * size.height,
      );
      canvas.drawRect(rect, boxPaint);
      _drawLabel(canvas, '${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%',
          Offset(rect.left, rect.top - 16), Colors.greenAccent);
    }

    final trackPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    for (final t in tracks) {
      final rect = Rect.fromLTRB(
        t.boundingBox[1] * size.width,
        t.boundingBox[0] * size.height,
        t.boundingBox[3] * size.width,
        t.boundingBox[2] * size.height,
      );
      canvas.drawRect(rect.deflate(2), trackPaint);
      _drawLabel(canvas, '#${t.trackId}', Offset(rect.right - 30, rect.top - 16), Colors.amberAccent);
    }
  }

  void _drawLabel(Canvas canvas, String text, Offset offset, Color color) {
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(color: color, fontSize: 12, backgroundColor: Colors.black54),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    painter.paint(canvas, offset);
  }

  @override
  bool shouldRepaint(covariant DetectionOverlayPainter oldDelegate) => true;
}