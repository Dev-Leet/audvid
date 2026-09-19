import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../../core/utils/logger.dart';

/// Tier D — sole purpose: detect a human body/skeleton. Nothing else.
///
/// KNOWN LIMITATION (stated plainly, not hidden): ML Kit Pose Detection
/// detects ONE prominent person per frame — it is not a multi-person
/// model. If you need simultaneous multi-person skeletons later, that
/// requires raw MediaPipe Tasks integration, a materially bigger lift with
/// no mature official Flutter plugin today.
class PoseDetectionService {
  PoseDetector? _detector;
  bool _isReady = false;
  bool get isLoaded => _isReady;

  Future<void> load() async {
    try {
      _detector = PoseDetector(
        options: PoseDetectorOptions(
          model: PoseDetectionModel.accurate, // matches "highly accurate" requirement
          mode: PoseDetectionMode.stream,      // optimized for continuous camera frames
        ),
      );
      _isReady = true;
      logger.info('PoseDetectionService', 'ML Kit Pose Detector initialized (accurate, stream mode).');
    } catch (e) {
      logger.error('PoseDetectionService', 'load() failed', e);
      _detector = null;
      _isReady = false;
    }
  }

  /// Returns the single detected pose, or null if none/failure.
  Future<Pose?> detect(InputImage inputImage) async {
    final detector = _detector;
    if (detector == null) return null;
    try {
      final poses = await detector
          .processImage(inputImage)
          .timeout(const Duration(seconds: 3));
      return poses.isEmpty ? null : poses.first;
    } catch (e) {
      logger.error('PoseDetectionService', 'detect() failed', e);
      return null;
    }
  }

  Future<void> dispose() async {
    try {
      await _detector?.close();
    } catch (e) {
      logger.warning('PoseDetectionService', 'close() failed', e);
    }
    _detector = null;
    _isReady = false;
  }
}