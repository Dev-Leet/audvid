import 'package:flutter/material.dart';
import '../core/models/evidence.dart';
import '../core/utils/json_pretty_printer.dart';

/// Renders one or more Evidence objects as live-updating, monospace JSON —
/// this IS the exact payload the real Risk Engine's Fusion Engine would
/// receive, so what's shown here is the ground truth for validation.
class EvidenceJsonPanel extends StatelessWidget {
  final String title;
  final List<Evidence> evidenceList;
  final Color? accentColor;

  const EvidenceJsonPanel({
    super.key,
    required this.title,
    required this.evidenceList,
    this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    final color = accentColor ?? Theme.of(context).colorScheme.primary;
    final jsonText = evidenceList.isEmpty
        ? '// No evidence yet — start the test to see live output.'
        : JsonPrettyPrinter.prettyList(evidenceList.map((e) => e.toJson()).toList());

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: color.withValues(alpha: 0.4)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(fontWeight: FontWeight.bold, color: color),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                Text('${evidenceList.length} item(s)',
                    style: TextStyle(fontSize: 12, color: color.withValues(alpha: 0.8))),
              ],
            ),
          ),
          Container(
            constraints: const BoxConstraints(maxHeight: 260),
            width: double.infinity,
            padding: const EdgeInsets.all(10),
            child: SingleChildScrollView(
              child: SelectableText(
                jsonText,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}