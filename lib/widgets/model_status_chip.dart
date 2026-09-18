import 'package:flutter/material.dart';

/// Green/red status chip — surfaces a failed .tflite load immediately on
/// the mode-select screen instead of discovering it deep inside a test.
class ModelStatusChip extends StatelessWidget {
  final String label;
  final bool isLoaded;

  const ModelStatusChip({super.key, required this.label, required this.isLoaded});

  @override
  Widget build(BuildContext context) {
    final color = isLoaded ? Colors.green : Colors.red;
    return Chip(
      avatar: Icon(isLoaded ? Icons.check_circle : Icons.error, color: color, size: 18),
      label: Text(label),
      backgroundColor: color.withOpacity(0.1),
      side: BorderSide(color: color.withOpacity(0.4)),
    );
  }
}