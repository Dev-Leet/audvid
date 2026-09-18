enum LogLevel { debug, info, warning, error }

class LogEntry {
  final DateTime timestamp;
  final LogLevel level;
  final String tag;
  final String message;
  final Object? error;

  LogEntry(this.timestamp, this.level, this.tag, this.message, [this.error]);
}

class TesterLogger {
  TesterLogger._();
  static final TesterLogger instance = TesterLogger._();

  final List<LogEntry> _buffer = [];
  static const int _maxBuffered = 300;

  void log(LogLevel level, String tag, String message, [Object? error]) {
    _buffer.add(LogEntry(DateTime.now(), level, tag, message, error));
    if (_buffer.length > _maxBuffered) _buffer.removeAt(0);
    // ignore: avoid_print
    print('[${level.name.toUpperCase()}] $tag: $message ${error ?? ''}');
  }

  void debug(String tag, String msg) => log(LogLevel.debug, tag, msg);
  void info(String tag, String msg) => log(LogLevel.info, tag, msg);
  void warning(String tag, String msg, [Object? e]) => log(LogLevel.warning, tag, msg, e);
  void error(String tag, String msg, [Object? e]) => log(LogLevel.error, tag, msg, e);

  List<LogEntry> recent(int n) =>
      _buffer.reversed.take(n).toList();
}

final logger = TesterLogger.instance;