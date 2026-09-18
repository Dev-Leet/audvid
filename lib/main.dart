import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/audio/yamnet_model.dart';
import 'services/vision/efficientdet_model.dart';
import 'services/vision/movinet_stream_model.dart';
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

  String _loadingText = 'Requesting permissions...';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    await Permission.microphone.request();
    await Permission.camera.request();

    // AUDIT FIX (T-14): the three loads previously ran SEQUENTIALLY with
    // no timeout — a single hung/corrupted .tflite asset would block the
    // loading screen indefinitely, denying access to the other two models
    // even if they were perfectly fine. Now loaded in PARALLEL, each with
    // an independent timeout, so one bad asset can't block the others.
    setState(() => _loadingText = 'Loading models in parallel...');

    await Future.wait([
      yamnet.load().timeout(const Duration(seconds: 15), onTimeout: () {
        // load() already catches its own errors internally and leaves
        // isLoaded == false; this timeout just prevents an indefinite hang
        // if Interpreter.fromAsset itself never returns.
      }),
      efficientDet.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
      movinet.load().timeout(const Duration(seconds: 15), onTimeout: () {}),
    ]);

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => ModeSelectScreen(
          yamnet: yamnet,
          efficientDet: efficientDet,
          movinet: movinet,
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