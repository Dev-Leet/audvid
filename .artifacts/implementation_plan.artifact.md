# Implementation Plan: NIRPADAM Model Tester (Android/Flutter)

The provided document (`Model_Tested_Initial_Documentation.txt`) contains a fully audited, comprehensive specification and source code for the "NIRPADAM Model Tester", a standalone app designed to validate the YAMNet and Vision (EfficientDet + MoViNet) models.

Although you requested an "Android application", the document provides a **Flutter/Dart codebase** (which natively compiles to Android). Since your current workspace (`audvid`) is already initialized as a Flutter project, I will extract, structure, and implement the complete Flutter codebase from the documentation to build the Android application.

## User Review Required

> [!IMPORTANT]
> **Native Android (Kotlin) vs. Flutter**
> The specification explicitly provides Dart/Flutter code. I will proceed with building the Flutter app (which runs on Android) based on the provided code blocks. If you strictly require this to be converted to a **Native Android (Kotlin/Jetpack Compose)** application instead, please let me know, and I will discard the Dart code and rewrite the entire architecture in Kotlin.

> [!WARNING]
> **Missing TFLite Assets**
> I will create the directory `assets/models/` and configure the project to load them, but I do not have access to the actual `.tflite` model files (`yamnet.tflite`, `efficientdet_lite0.tflite`, `movinet_a2_stream_int8.tflite`). You will need to manually place these three files into `assets/models/` before the app can successfully run.

## Proposed Changes

We will scaffold the Flutter project structure and extract all the code from the documentation, including the final "Audit Fixes" mentioned at the end of the text file.

### Configuration & Assets
- **[MODIFY]** `pubspec.yaml` - Add dependencies (`tflite_flutter`, `camera`, `record`, `permission_handler`, `battery_plus`, `image`, `path_provider`) and asset declarations.
- **[NEW]** `assets/config/coco_labelmap.txt`
- **[NEW]** `assets/config/yamnet_class_map.csv` (Using the fixed, quote-aware 521-class version)
- **[NEW]** `assets/config/kinetics600_labels.txt` (Using the full 600-class list)

### Core & Utils
- **[NEW]** `lib/core/models/evidence.dart`
- **[NEW]** `lib/core/models/test_config.dart`
- **[NEW]** `lib/core/utils/logger.dart`
- **[NEW]** `lib/core/utils/json_pretty_printer.dart`

### Services (Audio & Vision Models)
- **[NEW]** `lib/services/audio/mic_capture_service.dart` (With audit fixes for timeouts and WAV parsing)
- **[NEW]** `lib/services/audio/audio_preprocessing.dart`
- **[NEW]** `lib/services/audio/yamnet_model.dart` (With audit fixes for multi-output tensors and CSV parsing)
- **[NEW]** `lib/services/audio/audio_evidence_processor.dart`
- **[NEW]** `lib/services/vision/camera_controller_service.dart`
- **[NEW]** `lib/services/vision/vision_preprocessing.dart`
- **[NEW]** `lib/services/vision/efficientdet_model.dart`
- **[NEW]** `lib/services/vision/object_tracker.dart` (With audit fix for immutable snapshots)
- **[NEW]** `lib/services/vision/behavioral_feature_extractor.dart`
- **[NEW]** `lib/services/vision/movinet_stream_model.dart` (With audit fix for int8/float32 dtype validation)
- **[NEW]** `lib/services/vision/visual_evidence_processor.dart` (With audit fix replacing 'shouting' with 'arguing')

### Controllers
- **[NEW]** `lib/controllers/perf_stats.dart`
- **[NEW]** `lib/controllers/audio_test_controller.dart`
- **[NEW]** `lib/controllers/vision_test_controller.dart` (With critical T-2 audit fix preventing shared model disposal and T-3 reset cadence)
- **[NEW]** `lib/controllers/combined_test_controller.dart`

### UI (Screens & Widgets)
- **[NEW]** `lib/widgets/evidence_json_panel.dart`
- **[NEW]** `lib/widgets/audio_level_meter.dart`
- **[NEW]** `lib/widgets/detection_overlay_painter.dart`
- **[NEW]** `lib/widgets/model_status_chip.dart`
- **[NEW]** `lib/widgets/perf_stats_bar.dart`
- **[NEW]** `lib/screens/mode_select_screen.dart`
- **[NEW]** `lib/screens/audio_test_screen.dart`
- **[NEW]** `lib/screens/vision_test_screen.dart`
- **[NEW]** `lib/screens/combined_test_screen.dart`
- **[MODIFY]** `lib/main.dart` - Bootstrap models concurrently with timeouts.

## Verification Plan
1. Ensure `flutter pub get` runs successfully.
2. Confirm that there are no syntax or compilation errors in the Dart code.
3. Validate that the app boots into the `ModeSelectScreen` (even if models fail to load due to missing `.tflite` files, the UI should still render the failure chips).