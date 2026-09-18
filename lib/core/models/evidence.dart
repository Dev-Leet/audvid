/// Ported unchanged from the audited main NIRPADAM app. Keeping this
/// IDENTICAL is the whole point of this tester: the Evidence objects shown
/// here are exactly what the real Risk Engine would receive.
enum Modality {
  routeRisk,
  deviation,
  movement,
  timeContext,
  incidentContext,
  audio,
  vision,
  visionAction,
}

class Evidence {
  final Modality modality;
  final DateTime timestamp;
  final double confidence;
  final double riskContribution;
  final Map<String, dynamic> payload;
  final bool requiresCorroboration;
  final bool isUnavailable;
  final String? sourceKey;

  const Evidence({
    required this.modality,
    required this.timestamp,
    required this.confidence,
    required this.riskContribution,
    required this.payload,
    this.requiresCorroboration = false,
    this.isUnavailable = false,
    this.sourceKey,
  });

  Map<String, dynamic> toJson() => {
        'modality': modality.name,
        'timestamp': timestamp.toIso8601String(),
        'confidence': confidence,
        'risk_contribution': riskContribution,
        'requires_corroboration': requiresCorroboration,
        'is_unavailable': isUnavailable,
        'source_key': sourceKey,
        'payload': payload,
      };

  @override
  String toString() =>
      'Evidence(${modality.name}, risk=$riskContribution, conf=$confidence)';
}