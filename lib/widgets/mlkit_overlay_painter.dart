import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class MlkitOverlayPainter extends CustomPainter {
  final List<Face> faces;
  final Pose? pose;
  final Size? imageSize;
  final InputImageRotation? rotation;
  final CameraLensDirection? lensDirection;

  MlkitOverlayPainter({
    required this.faces,
    required this.pose,
    this.imageSize,
    this.rotation,
    this.lensDirection,
  });

  double translateX(double x, Size canvasSize) {
    if (imageSize == null || rotation == null || lensDirection == null) return x;
    final isRotated = rotation == InputImageRotation.rotation90deg || rotation == InputImageRotation.rotation270deg;
    // The image size provided by metadata is the unrotated sensor size (e.g. 1600x1200).
    // ML Kit returns coordinates relative to the ROTATED image (1200x1600).
    final scaledWidth = isRotated ? imageSize!.height : imageSize!.width;
    final scaleX = canvasSize.width / scaledWidth;
    
    // Front camera is horizontally mirrored by the preview
    if (lensDirection == CameraLensDirection.front) {
      return canvasSize.width - (x * scaleX);
    }
    return x * scaleX;
  }

  double translateY(double y, Size canvasSize) {
    if (imageSize == null || rotation == null || lensDirection == null) return y;
    final isRotated = rotation == InputImageRotation.rotation90deg || rotation == InputImageRotation.rotation270deg;
    final scaledHeight = isRotated ? imageSize!.width : imageSize!.height;
    final scaleY = canvasSize.height / scaledHeight;
    return y * scaleY;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final facePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.blueAccent;

    for (final face in faces) {
      final left = translateX(face.boundingBox.left, size);
      final top = translateY(face.boundingBox.top, size);
      final right = translateX(face.boundingBox.right, size);
      final bottom = translateY(face.boundingBox.bottom, size);

      final rect = Rect.fromLTRB(
        min(left, right),
        min(top, bottom),
        max(left, right),
        max(top, bottom),
      );
      
      canvas.drawRect(rect, facePaint);
      
      if (face.trackingId != null) {
        _drawLabel(canvas, 'ID: ${face.trackingId}', Offset(rect.left, rect.top - 16), Colors.blueAccent);
      }
    }

    if (pose != null) {
      final pointPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.orangeAccent;
        
      final linePaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = Colors.orangeAccent;

      pose!.landmarks.forEach((_, landmark) {
        canvas.drawCircle(Offset(translateX(landmark.x, size), translateY(landmark.y, size)), 3.0, pointPaint);
      });
      
      void drawLine(PoseLandmarkType t1, PoseLandmarkType t2) {
        final l1 = pose!.landmarks[t1];
        final l2 = pose!.landmarks[t2];
        if (l1 != null && l2 != null && l1.likelihood > 0.5 && l2.likelihood > 0.5) {
          canvas.drawLine(
            Offset(translateX(l1.x, size), translateY(l1.y, size)),
            Offset(translateX(l2.x, size), translateY(l2.y, size)),
            linePaint
          );
        }
      }

      drawLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow);
      drawLine(PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist);
      drawLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip);
      drawLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee);
      drawLine(PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle);
      
      drawLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow);
      drawLine(PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist);
      drawLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip);
      drawLine(PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee);
      drawLine(PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle);
      
      drawLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder);
      drawLine(PoseLandmarkType.leftHip, PoseLandmarkType.rightHip);
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