import 'package:flutter/material.dart';

/// Simple horizontal level meter driven by AudioTestController's RMS
/// stream — confirms the mic is actually capturing sound independent of
/// what YAMNet decides to classify it as.
class AudioLevelMeter extends StatelessWidget {
  final double level; // 0.0-1.0

  const AudioLevelMeter({super.key, required this.level});

  @override
  Widget build(BuildContext context) {
    final clamped = level.clamp(0.0, 1.0);
    final color = clamped > 0.7
        ? Colors.red
        : clamped > 0.3
            ? Colors.orange
            : Colors.green;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Mic Level', style: TextStyle(fontSize: 12, color: Colors.grey)),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: clamped,
            minHeight: 14,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
      ],
    );
  }
}