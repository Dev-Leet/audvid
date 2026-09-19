import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../../core/models/evidence.dart';

class FaceEvidenceProcessor {
  Evidence process(List<Face> faces, DateTime timestamp, int width, int height) {
    if (faces.isEmpty) {
      return Evidence(
        modality: Modality.vision,
        timestamp: timestamp,
        confidence: 0.6,
        riskContribution: 0, // detection-only tier — no scoring, by design
        payload: {
          'source': 'face_detector', 
          'face_count': 0,
          'resolution': '${width}x${height}'
        },
      );
    }

    final faceData = faces.map((f) => {
          // Session-scoped only (resets on detector restart) — NOT biometric
          // identity. This is the mechanism you chose.
          'tracking_id': f.trackingId,
          'bounding_box': {
            'left': f.boundingBox.left,
            'top': f.boundingBox.top,
            'right': f.boundingBox.right,
            'bottom': f.boundingBox.bottom,
          },
          'head_euler_angle_y': f.headEulerAngleY,
          'head_euler_angle_z': f.headEulerAngleZ,
        }).toList();

    return Evidence(
      modality: Modality.vision,
      timestamp: timestamp,
      confidence: 0.8,
      riskContribution: 0,
      payload: {
        'source': 'face_detector',
        'face_count': faces.length,
        'resolution': '${width}x${height}',
        'faces': faceData,
        'note': 'Detection-only evidence, no interpretation or risk scoring applied.',
      },
    );
  }
}