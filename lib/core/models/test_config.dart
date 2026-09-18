/// Trimmed-down configuration for the test harness. Deliberately excludes
/// every fusion/threshold/weight parameter from the main app's
/// RiskEngineConfig — this app never scores or fuses anything, so those
/// parameters have no meaning here.
class TesterConfig {
  final int audioWindowSeconds;
  final int visionBurstFrameCount;
  final int movinetWindowFrames;
  final int visionFrameSampleIntervalMs;
  final double behaviorHeuristicConfidenceCap;
  final double actionProxyConfidenceCap;
  final double minBatteryForVisionTierB; // for the synthetic battery-gate test

  // AUDIT FIX (T-15): combinedAudioIntervalSeconds/combinedVisionIntervalSeconds
  // were declared here but never read anywhere — CombinedTestController just
  // starts each sub-controller and relies on its own internal cadence.
  // Removed rather than left as dead config that silently ignores tuning.

  const TesterConfig({
    this.audioWindowSeconds = 2,
    this.visionBurstFrameCount = 8,
    this.movinetWindowFrames = 24,
    this.visionFrameSampleIntervalMs = 200,
    this.behaviorHeuristicConfidenceCap = 0.5,
    this.actionProxyConfidenceCap = 0.55,
    this.minBatteryForVisionTierB = 40.0,
  });

  static const defaults = TesterConfig();
}