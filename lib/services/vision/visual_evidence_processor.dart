import '../../core/models/evidence.dart';
import '../../core/utils/logger.dart';
import 'efficientdet_model.dart';
import 'movinet_stream_model.dart';

class VisualEvidenceProcessor {
  final double actionProxyConfidenceCap;

  VisualEvidenceProcessor({this.actionProxyConfidenceCap = 0.55});

  // Verified against the real, fetched kinetics_600_labels.txt — every key
  // below is confirmed present in that file. The original 'shouting' key
  // has been REMOVED: no such class exists in Kinetics-600 (confirmed by
  // checking the real label list), so it was silently dead the entire
  // time. 'arguing' is a real class and a more defensible verbal-conflict
  // proxy in its place.
  static const Map<String, double> _actionProxyWeights = {
    'wrestling': 55,
    'punching person (boxing)': 60,
    'arm wrestling': 20,
    'slapping': 50,
    'headbutting': 55,
    'arguing': 35,
  };

  String _normalizeLabel(String label) =>
      label.trim().toLowerCase().replaceAll(RegExp(r'\s+'), ' ');

  double? _lookupProxyWeight(String label) {
    final normalized = _normalizeLabel(label);
    if (_actionProxyWeights.containsKey(normalized)) return _actionProxyWeights[normalized];
    final stripped = normalized.replaceAll(RegExp(r'\s*\([^)]*\)'), '').trim();
    return _actionProxyWeights[stripped];
  }

  Evidence processDetections(List<Detection> detections, DateTime timestamp,
      {bool modelUnavailable = false}) {
    if (modelUnavailable) {
      return Evidence(
        modality: Modality.vision, timestamp: timestamp, confidence: 0,
        riskContribution: 0, isUnavailable: true,
        payload: {'status': 'detector_unavailable'},
      );
    }
    final personCount = detections.where((d) => d.label == 'person').length;
    final vehicleCount =
        detections.where((d) => ['car', 'motorcycle', 'bus', 'truck'].contains(d.label)).length;
    double score = 0;
    if (personCount == 0 && vehicleCount == 0) score = 5;
    if (personCount >= 4) score = 10;

    return Evidence(
      modality: Modality.vision, timestamp: timestamp,
      confidence: detections.isEmpty ? 0.3 : 0.7, riskContribution: score,
      payload: {
        'person_count': personCount, 'vehicle_count': vehicleCount,
        'detections': detections.map((d) => {'label': d.label, 'confidence': d.confidence}).toList(),
      },
    );
  }

  Evidence processAction(List<ActionClassResult> results, DateTime timestamp,
      {bool modelUnavailable = false}) {
    if (modelUnavailable || results.isEmpty) {
      return Evidence(
        modality: Modality.visionAction, timestamp: timestamp, confidence: 0,
        riskContribution: 0, isUnavailable: true,
        payload: {'status': modelUnavailable ? 'movinet_unavailable' : 'no_results'},
      );
    }

    final candidates = results.take(3).toList();
    double bestContribution = -1;
    ActionClassResult? bestMatch;
    double? bestWeight;

    for (final r in candidates) {
      final weight = _lookupProxyWeight(r.label);
      if (weight == null) continue;
      final contribution = weight * r.score;
      if (contribution > bestContribution) {
        bestContribution = contribution;
        bestMatch = r;
        bestWeight = weight;
      }
    }

    if (bestMatch == null) {
      final top = results.first;
      if (top.score > 0.5) {
        logger.debug('VisualEvidenceProcessor',
            'Unmapped high-confidence class "${top.label}" (${top.score.toStringAsFixed(2)})');
      }
      return Evidence(
        modality: Modality.visionAction, timestamp: timestamp, confidence: top.score,
        riskContribution: 0,
        payload: {
          'top_class': top.label, 'mapped_category': 'benign_activity',
          'note': 'Kinetics-600 class, not a purpose-built safety label.',
        },
      );
    }

    final cappedConfidence = bestMatch.score.clamp(0.0, actionProxyConfidenceCap);
    return Evidence(
      modality: Modality.visionAction, timestamp: timestamp, confidence: cappedConfidence,
      riskContribution: (bestWeight! * cappedConfidence).clamp(0, 100),
      requiresCorroboration: true,
      payload: {
        'top_class': bestMatch.label, 'mapped_category': 'physical_contact_proxy',
        'raw_score': bestMatch.score,
        'considered_top_n': candidates.map((r) => {'label': r.label, 'score': r.score}).toList(),
        'note': 'Kinetics-600 class mapped to a safety-adjacent proxy bucket; '
            'not a purpose-built violence classifier.',
      },
    );
  }
}