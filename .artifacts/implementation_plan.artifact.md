# Implementation Plan: Dynamic Camera Resolution

## User Request
The user wants to run the ML Kit models (Tier C & D) at maximum resolution without compressing the photo. However, Tier A & B (EfficientDet and MoViNet) will still run on their required, smaller resolutions. The user confirms they will *never* run Tier A/B simultaneously with Tier C/D.

## Answer
**Yes, this is 100% feasible!**

Because you are keeping them on separate test screens and never running them at the exact same time, we can simply configure the `CameraControllerService` to open the camera at `ResolutionPreset.medium` for Tier A/B, and close/re-open it at `ResolutionPreset.max` for Tier C/D.

## Proposed Changes

### `lib/services/vision/camera_controller_service.dart`
- **[MODIFY]** Update the `activate` method to accept an optional `ResolutionPreset` argument (defaulting to `ResolutionPreset.medium`).
- **[MODIFY]** Store the `ResolutionPreset` so that the correct resolution is requested when the camera initializes.

```dart
  Future<bool> activate({ResolutionPreset resolution = ResolutionPreset.medium}) async {
    // ...
    _controller = CameraController(cameras.first, resolution, enableAudio: false);
    // ...
  }
```

### `lib/controllers/vision_cd_test_controller.dart`
- **[MODIFY]** Update the call to `camera.activate()` to explicitly request maximum resolution since this screen only runs Tier C & D.

```dart
    final activated = await camera.activate(resolution: ResolutionPreset.max);
```

*(No changes are needed to `VisionTestController` or `CombinedTestController` because they will continue to use the default `ResolutionPreset.medium`.)*

## Verification Plan
1. Make the changes to `CameraControllerService` and `VisionCdTestController`.
2. Run `flutter analyze` to ensure no syntax errors.
3. Build the app and confirm the camera opens in high resolution on the Face/Pose screen, and standard resolution on the EfficientDet/MoViNet screen.