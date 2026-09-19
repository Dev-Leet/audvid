import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show WriteBuffer;
import 'package:google_mlkit_commons/google_mlkit_commons.dart';
import '../../core/utils/logger.dart';

class MlkitInputImageConverter {
  static InputImage? convert(
    CameraImage image, {
    required int sensorOrientation,
    required bool isFrontCamera,
    required DeviceOrientation deviceOrientation,
  }) {
    try {
      int deviceOrientationDegrees = 0;
      switch (deviceOrientation) {
        case DeviceOrientation.portraitUp:
          deviceOrientationDegrees = 0;
          break;
        case DeviceOrientation.landscapeLeft:
          deviceOrientationDegrees = 90;
          break;
        case DeviceOrientation.portraitDown:
          deviceOrientationDegrees = 180;
          break;
        case DeviceOrientation.landscapeRight:
          deviceOrientationDegrees = 270;
          break;
      }

      final rotationCompensation = isFrontCamera
          ? (sensorOrientation + deviceOrientationDegrees) % 360
          : (sensorOrientation - deviceOrientationDegrees + 360) % 360;

      final rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
      if (rotation == null) {
        logger.warning('MlkitInputImageConverter',
            'Unsupported rotation value: $rotationCompensation');
        return null;
      }

      // Convert all YUV420_888 images to strict NV21 to strip padding.
      // Padding/stride mismatch is the #1 cause of "0 faces detected" on Android at high resolutions.
      Uint8List bytes;
      InputImageFormat format;
      int bytesPerRow;

      if (image.format.raw == 35) { // 35 == ImageFormat.YUV_420_888 on Android
        bytes = _yuv420ToNv21(image);
        format = InputImageFormat.nv21;
        bytesPerRow = image.width; // NV21 is tightly packed, so bytesPerRow == width exactly
      } else {
        // iOS bgra8888 or pre-converted Android
        bytes = _concatenatePlanes(image.planes);
        format = InputImageFormatValue.fromRawValue(image.format.raw) ?? InputImageFormat.nv21;
        bytesPerRow = image.planes.first.bytesPerRow;
      }

      final metadata = InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: bytesPerRow,
      );

      return InputImage.fromBytes(bytes: bytes, metadata: metadata);
    } catch (e) {
      logger.error('MlkitInputImageConverter', 'convert() failed', e);
      return null;
    }
  }

  static Uint8List _concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (final plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  /// Manually converts Android Camera2 YUV_420_888 to tightly packed NV21.
  /// Eliminates invisible row-padding (stride) corruption at high resolutions
  /// which otherwise completely breaks ML Kit color channels and face detection.
  static Uint8List _yuv420ToNv21(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final ySize = width * height;
    final uvSize = width * height ~/ 2;
    final nv21 = Uint8List(ySize + uvSize);

    // Copy Y plane, safely ignoring padding
    final yRowStride = yPlane.bytesPerRow;
    if (yRowStride == width) {
      nv21.setRange(0, ySize, yPlane.bytes);
    } else {
      var offset = 0;
      for (var i = 0; i < height; i++) {
        nv21.setRange(offset, offset + width,
            yPlane.bytes.sublist(i * yRowStride, i * yRowStride + width));
        offset += width;
      }
    }

    // Copy V and U planes (NV21 expects V then U interleaved)
    final uvRowStride = vPlane.bytesPerRow;
    final uvPixelStride = vPlane.bytesPerPixel ?? 1;

    var nv21Index = ySize;
    for (var row = 0; row < height ~/ 2; row++) {
      var vOffset = row * uvRowStride;
      var uOffset = row * uPlane.bytesPerRow;
      for (var col = 0; col < width ~/ 2; col++) {
        nv21[nv21Index++] = vPlane.bytes[vOffset];
        nv21[nv21Index++] = uPlane.bytes[uOffset];
        vOffset += uvPixelStride;
        uOffset += uPlane.bytesPerPixel ?? 1;
      }
    }
    return nv21;
  }
}