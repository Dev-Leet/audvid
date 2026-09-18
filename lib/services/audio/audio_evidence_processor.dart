import '../../core/models/evidence.dart';
import 'yamnet_model.dart';

class AudioEvidenceProcessor {
  final double confidenceCapUnused = 1.0; // no fusion in the tester — kept for parity

  static const Map<String, double> _highSeverityWeights = {
    'Screaming': 80, 'Gunshot, gunfire': 95, 'Explosion': 90,
    'Glass': 40, 'Crying, sobbing': 55, 'Shout': 45, 'Yell': 45,
  };

  static const Map<String, double> _ambientContextWeights = {
    'Siren': 30, 'Alarm': 25, 'Vehicle horn, car horn, honking': 10,
    'Traffic noise, roadway noise': 5, 'Crowd': 15, 'Run': 15,
  };

  Evidence process(List<AudioClassResult> results, DateTime timestamp) {
    if (results.isEmpty) {
      return Evidence(
        modality: Modality.audio,
        timestamp: timestamp,
        confidence: 0.0,
        riskContribution: 0,
        isUnavailable: true,
        payload: {'status': 'no_results'},
      );
    }

    double maxRisk = 0;
    AudioClassResult? topMatch;
    bool isHighSeverity = false;

    for (final r in results) {
      final highWeight = _highSeverityWeights[r.label];
      final weight = highWeight ?? _ambientContextWeights[r.label];
      if (weight == null) continue;
      final contribution = weight * r.score;
      if (contribution > maxRisk) {
        maxRisk = contribution;
        topMatch = r;
        isHighSeverity = highWeight != null;
      }
    }

    if (topMatch == null) {
      return Evidence(
        modality: Modality.audio,
        timestamp: timestamp,
        confidence: results.first.score,
        riskContribution: 0,
        payload: {
          'top_events': results.take(5).map((r) => {'label': r.label, 'score': r.score}).toList(),
          'matched_category': null,
        },
      );
    }

    return Evidence(
      modality: Modality.audio,
      timestamp: timestamp,
      confidence: topMatch.score,
      riskContribution: maxRisk.clamp(0, 100),
      requiresCorroboration: !isHighSeverity,
      payload: {
        'top_events': results.take(5).map((r) => {'label': r.label, 'score': r.score}).toList(),
        'matched_category': topMatch.label,
        'severity_tier': isHighSeverity ? 'high' : 'ambient',
      },
    );
  }
}