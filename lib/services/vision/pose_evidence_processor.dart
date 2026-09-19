import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../../core/models/evidence.dart';

class PoseEvidenceProcessor {
  Evidence process(Pose? pose, DateTime timestamp, int width, int height) {
    if (pose == null || pose.landmarks.isEmpty) {
      return Evidence(
        modality: Modality.vision,
        timestamp: timestamp,
        confidence: 0.5,
        riskContribution: 0,
        payload: {
          'source': 'pose_detector', 
          'person_detected': false,
          'resolution': '${width}x${height}'
        },
      );
    }

    final landmarks = pose.landmarks.map((type, lm) => MapEntry(type.name, {
          'x': lm.x,
          'y': lm.y,
          'z': lm.z,
          'likelihood': lm.likelihood,
        }));

    final avgLikelihood =
        pose.landmarks.values.map((l) => l.likelihood).reduce((a, b) => a + b) /
            pose.landmarks.length;

    return Evidence(
      modality: Modality.vision,
      timestamp: timestamp,
      confidence: avgLikelihood,
      riskContribution: 0, // detection-only tier, matching FaceEvidenceProcessor
      payload: {
        'source': 'pose_detector',
        'person_detected': true,
        'resolution': '${width}x${height}',
        'landmark_count': pose.landmarks.length,
        'avg_landmark_confidence': avgLikelihood,
        'landmarks': landmarks,
        'note': 'Single-person only (ML Kit limitation) — see class doc.',
      },
    );
  }
}