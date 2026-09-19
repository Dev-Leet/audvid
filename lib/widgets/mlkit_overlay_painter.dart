import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class MlkitOverlayPainter extends CustomPainter {
  final List<Face> faces;
  final Pose? pose;

  MlkitOverlayPainter({required this.faces, required this.pose});

  @override
  void paint(Canvas canvas, Size size) {
    // Note: This draws the raw bounding boxes assuming the camera preview
    // is completely unscaled/unrotated. In a true production app, you would
    // apply an affine transform here to map the image coordinates to the
    // screen coordinates (considering crop, scale, and device rotation).
    // For this tester, it provides enough visual confirmation that the models
    // are firing.
    
    // Draw Faces
    final facePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.blueAccent;

    for (final face in faces) {
      // Scale bounding box to canvas size
      // ML Kit usually outputs coordinates matching the absolute image resolution.
      // E.g., if the image is 480x640, coordinates will be in that range.
      // We do a naive scale here for the tester UI.
      final rect = Rect.fromLTRB(
        face.boundingBox.left,
        face.boundingBox.top,
        face.boundingBox.right,
        face.boundingBox.bottom,
      );
      
      canvas.drawRect(rect, facePaint);
      
      if (face.trackingId != null) {
        _drawLabel(canvas, 'ID: ${face.trackingId}', Offset(rect.left, rect.top - 16), Colors.blueAccent);
      }
    }

    // Draw Pose
    if (pose != null) {
      final pointPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.orangeAccent;
        
      final linePaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = Colors.orangeAccent;

      // Draw all landmarks
      pose!.landmarks.forEach((_, landmark) {
        canvas.drawCircle(Offset(landmark.x, landmark.y), 3.0, pointPaint);
      });
      
      // Draw a few key skeleton lines (Left side)
      _drawLine(canvas, pose!, PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, linePaint);
      
      // Right side
      _drawLine(canvas, pose!, PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, linePaint);
      
      // Center
      _drawLine(canvas, pose!, PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder, linePaint);
      _drawLine(canvas, pose!, PoseLandmarkType.leftHip, PoseLandmarkType.rightHip, linePaint);
    }
  }
  
  void _drawLine(Canvas canvas, Pose pose, PoseLandmarkType t1, PoseLandmarkType t2, Paint paint) {
    final l1 = pose.landmarks[t1];
    final l2 = pose.landmarks[t2];
    if (l1 != null && l2 != null && l1.likelihood > 0.5 && l2.likelihood > 0.5) {
      canvas.drawLine(Offset(l1.x, l1.y), Offset(l2.x, l2.y), paint);
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
  bool shouldRepaint(covariant MlkitOverlayPainter oldDelegate) => true;
}