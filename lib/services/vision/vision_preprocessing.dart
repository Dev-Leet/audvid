import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

class _IsolateData {
  final int width;
  final int height;
  final int yBytesPerRow;
  final int uBytesPerRow;
  final int vBytesPerRow;
  final int uPixelStride;
  final int vPixelStride;
  final Uint8List yPlane;
  final Uint8List uPlane;
  final Uint8List vPlane;
  final int targetSize;
  final int rotationDegrees;

  _IsolateData({
    required this.width,
    required this.height,
    required this.yBytesPerRow,
    required this.uBytesPerRow,
    required this.vBytesPerRow,
    required this.uPixelStride,
    required this.vPixelStride,
    required this.yPlane,
    required this.uPlane,
    required this.vPlane,
    required this.targetSize,
    required this.rotationDegrees,
  });
}

class VisionPreprocessing {
  static const int efficientDetInputSize = 320;
  static const int movinetInputSize = 172;

  static Future<Uint8List> cameraImageToRgbBytesIsolate(CameraImage image, int targetSize, {int rotationDegrees = 0}) async {
    final data = _IsolateData(
      width: image.width,
      height: image.height,
      yBytesPerRow: image.planes[0].bytesPerRow,
      uBytesPerRow: image.planes[1].bytesPerRow,
      vBytesPerRow: image.planes[2].bytesPerRow,
      uPixelStride: image.planes[1].bytesPerPixel ?? 1,
      vPixelStride: image.planes[2].bytesPerPixel ?? 1,
      yPlane: image.planes[0].bytes,
      uPlane: image.planes[1].bytes,
      vPlane: image.planes[2].bytes,
      targetSize: targetSize,
      rotationDegrees: rotationDegrees,
    );

    return Isolate.run(() => _processInIsolate(data));
  }

  static Uint8List _processInIsolate(_IsolateData data) {
    var converted = _convertYuv420ToImage(data);
    
    if (data.rotationDegrees != 0) {
      converted = img.copyRotate(converted, angle: data.rotationDegrees);
    }
    
    final resized = img.copyResize(converted,
        width: data.targetSize, height: data.targetSize, interpolation: img.Interpolation.linear);
        
    final out = Uint8List(data.targetSize * data.targetSize * 3);
    var idx = 0;
    for (var y = 0; y < data.targetSize; y++) {
      for (var x = 0; x < data.targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        out[idx++] = pixel.r.toInt();
        out[idx++] = pixel.g.toInt();
        out[idx++] = pixel.b.toInt();
      }
    }
    return out;
  }

  static img.Image _convertYuv420ToImage(_IsolateData data) {
    final width = data.width;
    final height = data.height;
    
    final maxYIndex = (height - 1) * data.yBytesPerRow + (width - 1);
    final maxUvRow = (height ~/ 2) - 1;
    final maxUvCol = (width ~/ 2) - 1;
    final maxUIndex = maxUvRow * data.uBytesPerRow + maxUvCol * data.uPixelStride;
    final maxVIndex = maxUvRow * data.vBytesPerRow + maxUvCol * data.vPixelStride;
    
    if (maxYIndex >= data.yPlane.length ||
        maxUIndex >= data.uPlane.length ||
        maxVIndex >= data.vPlane.length) {
      throw StateError('YUV plane bounds check failed — layout mismatch.');
    }

    final out = img.Image(width: width, height: height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIndex = y * data.yBytesPerRow + x;
        final uvRow = y ~/ 2;
        final uvCol = x ~/ 2;
        final uIndex = uvRow * data.uBytesPerRow + uvCol * data.uPixelStride;
        final vIndex = uvRow * data.vBytesPerRow + uvCol * data.vPixelStride;
        
        final yVal = data.yPlane[yIndex];
        final uVal = data.uPlane[uIndex] - 128;
        final vVal = data.vPlane[vIndex] - 128;
        
        final r = (yVal + 1.370705 * vVal).clamp(0, 255).toInt();
        final g = (yVal - 0.337633 * uVal - 0.698001 * vVal).clamp(0, 255).toInt();
        final b = (yVal + 1.732446 * uVal).clamp(0, 255).toInt();
        out.setPixelRgb(x, y, r, g, b);
      }
    }
    return out;
  }
}
