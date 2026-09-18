import 'package:flutter/material.dart';
import '../controllers/perf_stats.dart';

/// Shows rolling inference latency / failure-rate stats — the same numbers
/// the main-app audit flagged as needing verification (frame throttling,
/// timeout behavior) are visible live here.
class PerfStatsBar extends StatelessWidget {
  final String label;
  final PerfStats stats;

  const PerfStatsBar({super.key, required this.label, required this.stats});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 12)),
          Text('avg ${stats.avgLatencyMs.toStringAsFixed(0)}ms · last ${stats.lastLatencyMs}ms',
              style: const TextStyle(fontSize: 12)),
          Text('${stats.totalCycles} cycles · ${(stats.failureRate * 100).toStringAsFixed(0)}% failed',
              style: TextStyle(fontSize: 12, color: stats.failureRate > 0.2 ? Colors.red : Colors.grey)),
        ],
      ),
    );
  }
}