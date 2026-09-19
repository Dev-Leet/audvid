import 'dart:async';
import 'package:flutter/material.dart';
import '../services/audio/yamnet_model.dart';
import '../services/vision/efficientdet_model.dart';
import '../services/vision/movinet_stream_model.dart';
import '../services/vision/face_detection_service.dart';
import '../services/vision/pose_detection_service.dart';
import '../widgets/model_status_chip.dart';
import 'audio_test_screen.dart';
import 'vision_test_screen.dart';
import 'vision_cd_test_screen.dart';
import 'combined_test_screen.dart';

class ModeSelectScreen extends StatefulWidget {
  final YamnetModel yamnet;
  final EfficientDetModel efficientDet;
  final MovinetStreamModel movinet;
  final FaceDetectionService faceService;
  final PoseDetectionService poseService;

  const ModeSelectScreen({
    super.key,
    required this.yamnet,
    required this.efficientDet,
    required this.movinet,
    required this.faceService,
    required this.poseService,
  });

  @override
  State<ModeSelectScreen> createState() => _ModeSelectScreenState();
}

class _ModeSelectScreenState extends State<ModeSelectScreen> with WidgetsBindingObserver {
  Timer? _statusPollTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startPolling();
  }

  void _startPolling() {
    _statusPollTimer?.cancel();
    _statusPollTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {}); // re-reads isLoaded getters on every tick
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Pause polling when backgrounded, resume when foregrounded again.
    if (state == AppLifecycleState.resumed) {
      _startPolling();
    } else {
      _statusPollTimer?.cancel();
    }
  }

  @override
  void dispose() {
    _statusPollTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NIRPADAM Model Tester')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Model Load Status', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ModelStatusChip(label: 'YAMNet (Audio)', isLoaded: widget.yamnet.isLoaded),
                ModelStatusChip(
                    label: 'EfficientDet-Lite0 (Vision Tier A)', isLoaded: widget.efficientDet.isLoaded),
                ModelStatusChip(
                    label: 'MoViNet-Stream (Vision Tier B / Option 2)', isLoaded: widget.movinet.isLoaded),
                ModelStatusChip(
                    label: 'ML Kit Face (Vision Tier C)', isLoaded: widget.faceService.isLoaded),
                ModelStatusChip(
                    label: 'ML Kit Pose (Vision Tier D)', isLoaded: widget.poseService.isLoaded),
              ],
            ),
            const SizedBox(height: 32),
            const Text(
              'This app tests each model in isolation and shows the exact '
              'Evidence JSON that would be sent to the real Risk Engine — '
              'no fusion or scoring happens here.',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 32),
            _ModeButton(
              icon: Icons.mic,
              label: 'Audio Test (YAMNet only)',
              enabled: widget.yamnet.isLoaded,
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => AudioTestScreen(yamnet: widget.yamnet)),
              ).then((_) { if (mounted) setState(() {}); }), // re-check status immediately on return
            ),
            const SizedBox(height: 12),
            _ModeButton(
              icon: Icons.camera_alt,
              label: 'Vision Test (EfficientDet + MoViNet)',
              enabled: widget.efficientDet.isLoaded || widget.movinet.isLoaded,
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) =>
                        VisionTestScreen(efficientDet: widget.efficientDet, movinet: widget.movinet)),
              ).then((_) { if (mounted) setState(() {}); }),
            ),
            const SizedBox(height: 12),
            _ModeButton(
              icon: Icons.face,
              label: 'Vision Test (Face & Pose ONLY)',
              enabled: widget.faceService.isLoaded || widget.poseService.isLoaded,
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (_) =>
                        VisionCdTestScreen(faceService: widget.faceService, poseService: widget.poseService)),
              ).then((_) { if (mounted) setState(() {}); }),
            ),
            const SizedBox(height: 12),
            _ModeButton(
              icon: Icons.merge_type,
              label: 'Combined Test (Audio + Vision concurrently)',
              enabled: widget.yamnet.isLoaded && (widget.efficientDet.isLoaded || widget.movinet.isLoaded),
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => CombinedTestScreen(
                    yamnet: widget.yamnet,
                    efficientDet: widget.efficientDet,
                    movinet: widget.movinet,
                  ),
                ),
              ).then((_) { if (mounted) setState(() {}); }),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModeButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool enabled;
  final VoidCallback onTap;

  const _ModeButton({
    required this.icon,
    required this.label,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      icon: Icon(icon),
      label: Text(label),
      onPressed: enabled ? onTap : null,
      style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
    );
  }
}