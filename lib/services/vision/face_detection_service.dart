import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../../core/utils/logger.dart';

/// Tier C — sole purpose: detect human faces. Nothing else.
///
/// `enableTracking: true` gives each detected face a session-scoped
/// `trackingId` — this IS the "session-only tracking ID, no biometrics"
/// mechanism you chose. It resets whenever the detector is closed/reopened
/// and never persists identity across app restarts or camera sessions.
/// `performanceMode: accurate` is used over `fast`, matching the explicit
/// "highly accurate" requirement over raw speed.
class FaceDetectionService {
  FaceDetector? _detector;
  bool _isReady = false;
  bool get isLoaded => _isReady;

  Future<void> load() async {
    try {
      _detector = FaceDetector(
        options: FaceDetectorOptions(
          enableTracking: true,
          enableLandmarks: true,
          enableContours: false, // heavier, not needed for detection-only evidence
          enableClassification: false, // smiling/eyes-open are out of scope
          performanceMode: FaceDetectorMode.accurate,
        ),
      );
      _isReady = true;
      logger.info('FaceDetectionService', 'ML Kit Face Detector initialized (accurate mode, tracking on).');
    } catch (e) {
      logger.error('FaceDetectionService', 'load() failed', e);
      _detector = null;
      _isReady = false;
    }
  }

  /// Guarded with a timeout — this is a platform-channel call into native
  /// code, and per the whole codebase's established C-1/C-2 discipline, no
  /// external/native call goes unguarded.
  Future<List<Face>> detect(InputImage inputImage) async {
    final detector = _detector;
    if (detector == null) return const [];
    try {
      return await detector
          .processImage(inputImage)
          .timeout(const Duration(seconds: 3));
    } catch (e) {
      logger.error('FaceDetectionService', 'detect() failed', e);
      return const [];
    }
  }

  Future<void> dispose() async {
    try {
      await _detector?.close();
    } catch (e) {
      logger.warning('FaceDetectionService', 'close() failed', e);
    }
    _detector = null;
    _isReady = false;
  }
}