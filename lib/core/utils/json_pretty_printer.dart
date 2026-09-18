import 'dart:convert';

/// Pretty-prints an Evidence.toJson() map (or any JSON-serializable map)
/// for display in the live inspector panels. Kept dependency-free —
/// no external JSON-viewer package needed for a monospace text render.
class JsonPrettyPrinter {
  static String pretty(Map<String, dynamic> json) {
    const encoder = JsonEncoder.withIndent('  ');
    try {
      return encoder.convert(json);
    } catch (e) {
      return json.toString();
    }
  }

  static String prettyList(List<Map<String, dynamic>> items) {
    const encoder = JsonEncoder.withIndent('  ');
    try {
      return encoder.convert(items);
    } catch (e) {
      return items.toString();
    }
  }
}