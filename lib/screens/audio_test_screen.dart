import 'package:flutter/material.dart';
import '../controllers/audio_test_controller.dart';
import '../core/models/evidence.dart';
import '../services/audio/yamnet_model.dart';
import '../services/audio/audio_evidence_processor.dart';
import '../services/audio/mic_capture_service.dart';
import '../widgets/audio_level_meter.dart';
import '../widgets/evidence_json_panel.dart';
import '../widgets/perf_stats_bar.dart';

class AudioTestScreen extends StatefulWidget {
  final YamnetModel yamnet;
  const AudioTestScreen({super.key, required this.yamnet});

  @override
  State<AudioTestScreen> createState() => _AudioTestScreenState();
}

class _AudioTestScreenState extends State<AudioTestScreen> {
  late final AudioTestController controller;

  List<AudioClassResult> _rawResults = [];
  List<Evidence> _evidenceHistory = [];
  double _level = 0.0;
  String _status = 'Idle';

  @override
  void initState() {
    super.initState();
    controller = AudioTestController(
      model: widget.yamnet,
      processor: AudioEvidenceProcessor(),
      micCapture: MicCaptureService(),
    );
    controller.rawResultsStream.listen((r) { if (mounted) setState(() => _rawResults = r); });
    controller.evidenceStream.listen((e) { if (mounted) setState(() {
          _evidenceHistory = [e, ..._evidenceHistory].take(10).toList();
        }); });
    controller.levelStream.listen((l) { if (mounted) setState(() => _level = l); });
    controller.statusStream.listen((s) { if (mounted) setState(() => _status = s); });
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Audio Test — YAMNet')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(child: Text('Status: $_status')),
                ElevatedButton(
                  onPressed: controller.isRunning ? controller.stop : controller.start,
                  style: ElevatedButton.styleFrom(
                      backgroundColor: controller.isRunning ? Colors.red : Colors.green),
                  child: Text(controller.isRunning ? 'Stop' : 'Start'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            AudioLevelMeter(level: _level),
            const SizedBox(height: 12),
            PerfStatsBar(label: 'YAMNet inference', stats: controller.perfStats),
            const SizedBox(height: 16),
            const Text('Top Class Scores (raw model output)',
                style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            SizedBox(
              height: 160,
              child: ListView.builder(
                itemCount: _rawResults.length,
                itemBuilder: (context, i) {
                  final r = _rawResults[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      children: [
                        SizedBox(width: 160, child: Text(r.label, overflow: TextOverflow.ellipsis)),
                        Expanded(
                          child: LinearProgressIndicator(value: r.score.clamp(0.0, 1.0)),
                        ),
                        SizedBox(width: 50, child: Text(' ${(r.score * 100).toStringAsFixed(1)}%')),
                      ],
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: EvidenceJsonPanel(
                title: 'Evidence sent to Risk Engine (last 10 windows)',
                evidenceList: _evidenceHistory,
                accentColor: Colors.blue,
              ),
            ),
          ],
        ),
      ),
    );
  }
}