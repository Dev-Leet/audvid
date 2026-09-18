/// Simple rolling performance stats — inference latency and cycle counts,
/// shown in the perf_stats_bar widget on every test screen so you can spot
/// a model that's too slow for real-time use before it ships.
class PerfStats {
  final List<int> _latenciesMs = [];
  int totalCycles = 0;
  int failedCycles = 0;
  DateTime? lastRunAt;

  static const int _maxSamples = 20;

  void recordLatency(int ms) {
    _latenciesMs.add(ms);
    if (_latenciesMs.length > _maxSamples) _latenciesMs.removeAt(0);
    totalCycles++;
    lastRunAt = DateTime.now();
  }

  void recordFailure() {
    failedCycles++;
    totalCycles++;
    lastRunAt = DateTime.now();
  }

  double get avgLatencyMs =>
      _latenciesMs.isEmpty ? 0 : _latenciesMs.reduce((a, b) => a + b) / _latenciesMs.length;

  int get lastLatencyMs => _latenciesMs.isEmpty ? 0 : _latenciesMs.last;

  double get failureRate => totalCycles == 0 ? 0 : failedCycles / totalCycles;

  Map<String, dynamic> toJson() => {
        'avg_latency_ms': avgLatencyMs.toStringAsFixed(0),
        'last_latency_ms': lastLatencyMs,
        'total_cycles': totalCycles,
        'failed_cycles': failedCycles,
        'failure_rate_pct': (failureRate * 100).toStringAsFixed(1),
      };
}