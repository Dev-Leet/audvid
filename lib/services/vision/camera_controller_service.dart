import 'dart:async';
import 'package:camera/camera.dart';
import '../../core/utils/logger.dart';

class CameraControllerService {
  CameraController? _controller;
  CameraImage? _latestFrame;

  bool get isActive => _controller?.value.isInitialized ?? false;
  int? get previewHeightPx => _controller?.value.previewSize?.height.toInt();
  int? get previewWidthPx => _controller?.value.previewSize?.width.toInt();
  CameraController? get rawController => _controller; // exposed for the live preview widget

  Future<bool> activate({ResolutionPreset resolution = ResolutionPreset.medium}) async {
    try {
      final cameras = await availableCameras().timeout(
        const Duration(seconds: 5),
        onTimeout: () => throw TimeoutException('availableCameras() timed out'),
      );
      if (cameras.isEmpty) {
        logger.warning('CameraControllerService', 'No cameras available');
        return false;
      }
      _controller = CameraController(cameras.first, resolution, enableAudio: false);
      await _controller!.initialize().timeout(
        const Duration(seconds: 8),
        onTimeout: () => throw TimeoutException('Camera initialize() timed out'),
      );
      _latestFrame = null;
      await _controller!.startImageStream((image) => _latestFrame = image);
      return true;
    } catch (e) {
      logger.error('CameraControllerService', 'activate() failed', e);
      _controller = null;
      _latestFrame = null;
      return false;
    }
  }

  Future<CameraImage?> waitForNextFrame({Duration timeout = const Duration(seconds: 2)}) async {
    if (!isActive) return null;
    final deadline = DateTime.now().add(timeout);
    final startingFrame = _latestFrame;
    while (DateTime.now().isBefore(deadline)) {
      if (_latestFrame != null && !identical(_latestFrame, startingFrame)) return _latestFrame;
      await Future.delayed(const Duration(milliseconds: 20));
    }
    logger.warning('CameraControllerService', 'waitForNextFrame() timed out');
    return _latestFrame;
  }

  Future<void> deactivate() async {
    try {
      await _controller?.stopImageStream().timeout(const Duration(seconds: 3));
    } catch (e) {
      logger.warning('CameraControllerService', 'stopImageStream() failed', e);
    }
    try {
      await _controller?.dispose();
    } catch (e) {
      logger.warning('CameraControllerService', 'dispose() failed', e);
    }
    _controller = null;
    _latestFrame = null;
  }
}