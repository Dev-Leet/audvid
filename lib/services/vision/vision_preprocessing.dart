import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

class VisionPreprocessing {
  static const int efficientDetInputSize = 320;
  static const int movinetInputSize = 172;

  static Uint8List cameraImageToRgbBytes(CameraImage image, int targetSize) {
    final converted = _convertYuv420ToImage(image);
    final resized = img.copyResize(converted,
        width: targetSize, height: targetSize, interpolation: img.Interpolation.linear);
    final out = Uint8List(targetSize * targetSize * 3);
    var idx = 0;
    for (var y = 0; y < targetSize; y++) {
      for (var x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        out[idx++] = pixel.r.toInt();
        out[idx++] = pixel.g.toInt();
        out[idx++] = pixel.b.toInt();
      }
    }
    return out;
  }

  static img.Image _convertYuv420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    if (image.planes.length < 3) {
      throw StateError('Expected 3 YUV planes, got ${image.planes.length}.');
    }
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final uPixelStride = uPlane.bytesPerPixel ?? 1;
    final vPixelStride = vPlane.bytesPerPixel ?? 1;

    final maxYIndex = (height - 1) * yPlane.bytesPerRow + (width - 1);
    final maxUvRow = (height ~/ 2) - 1;
    final maxUvCol = (width ~/ 2) - 1;
    final maxUIndex = maxUvRow * uPlane.bytesPerRow + maxUvCol * uPixelStride;
    final maxVIndex = maxUvRow * vPlane.bytesPerRow + maxUvCol * vPixelStride;
    if (maxYIndex >= yPlane.bytes.length ||
        maxUIndex >= uPlane.bytes.length ||
        maxVIndex >= vPlane.bytes.length) {
      throw StateError('YUV plane bounds check failed — layout mismatch.');
    }

    final out = img.Image(width: width, height: height);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIndex = y * yPlane.bytesPerRow + x;
        final uvRow = y ~/ 2;
        final uvCol = x ~/ 2;
        final uIndex = uvRow * uPlane.bytesPerRow + uvCol * uPixelStride;
        final vIndex = uvRow * vPlane.bytesPerRow + uvCol * vPixelStride;
        final yVal = yPlane.bytes[yIndex];
        final uVal = uPlane.bytes[uIndex] - 128;
        final vVal = vPlane.bytes[vIndex] - 128;
        final r = (yVal + 1.370705 * vVal).clamp(0, 255).toInt();
        final g = (yVal - 0.337633 * uVal - 0.698001 * vVal).clamp(0, 255).toInt();
        final b = (yVal + 1.732446 * uVal).clamp(0, 255).toInt();
        out.setPixelRgb(x, y, r, g, b);
      }
    }
    return out;
  }
}