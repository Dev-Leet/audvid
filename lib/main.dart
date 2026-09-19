import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/audio/yamnet_model.dart';
import 'services/vision/efficientdet_model.dart';
import 'services/vision/movinet_stream_model.dart';
import 'services/vision/face_detection_service.dart';
import 'services/vision/pose_detection_service.dart';
import 'screens/mode_select_screen.dart';

void main() {
  runApp(const ModelTesterApp());
}

class ModelTesterApp extends StatelessWidget {
  const ModelTesterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NIRPADAM Model Tester',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const _BootstrapScreen(),
    );
  }
}

class _BootstrapScreen extends StatefulWidget {
  const _BootstrapScreen();

  @override
  State<_BootstrapScreen> createState() => _BootstrapScreenState();
}

class _BootstrapScreenState extends State<_BootstrapScreen> {
  final yamnet = YamnetModel();
  final efficientDet = EfficientDetModel();
  final movinet = MovinetStreamModel();
  final faceService = FaceDetectionService();
  final poseService = PoseDetectionService();

  String _loadingText = 'Requesting permissions...';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    await Permission.microphone.request();
    await Permission.camera.request();

    setState(() => _loadingText = 'Loading all models in parallel...');

    await Future.wait([
      yamnet.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
      efficientDet.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
      movinet.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
      faceService.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
      poseService.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
    ]);

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => ModeSelectScreen(
          yamnet: yamnet,
          efficientDet: efficientDet,
          movinet: movinet,
          faceService: faceService,
          poseService: poseService,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 20),
            Text(_loadingText),
          ],
        ),
      ),
    );
  }
}