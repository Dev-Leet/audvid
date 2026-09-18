import '../../core/models/evidence.dart';
import 'object_tracker.dart';

class _HeadingBearingSample {
  final DateTime timestamp;
  final double userHeadingDegrees;
  final double objectAbsoluteBearingDegrees;
  _HeadingBearingSample(this.timestamp, this.userHeadingDegrees, this.objectAbsoluteBearingDegrees);
}

class BehavioralFeatureExtractor {
  static const double _assumedPersonHeightMeters = 1.7;
  static const double _stationaryThresholdMps = 0.3;
  static const double _turnThresholdDegrees = 25.0;
  static const double _bearingStabilityToleranceDegrees = 15.0;

  final double behaviorHeuristicConfidenceCap;
  final Map<int, List<_HeadingBearingSample>> _headingHistory = {};
  final List<double> _recentUserSpeeds = [];

  BehavioralFeatureExtractor({this.behaviorHeuristicConfidenceCap = 0.5});

  List<Evidence> extract({
    required List<TrackedObject> tracks,
    required double cameraFocalLengthPx,
    required int frameHeightPx,
    required double cameraHorizontalFovDegrees,
    required double? userHeadingDegrees,
    required double? userSpeedMetersPerSecond,
    required DateTime timestamp,
  }) {
    if (userSpeedMetersPerSecond != null) {
      _recentUserSpeeds.add(userSpeedMetersPerSecond);
      if (_recentUserSpeeds.length > 6) _recentUserSpeeds.removeAt(0);
    }

    final evidence = <Evidence>[];
    evidence.addAll(_detectPossibleFollowing(
      tracks, cameraFocalLengthPx, frameHeightPx, cameraHorizontalFovDegrees,
      userHeadingDegrees, timestamp,
    ));
    evidence.addAll(_detectBlockedPath(tracks, userSpeedMetersPerSecond, timestamp));
    evidence.addAll(_detectCloseStaticInteraction(tracks, timestamp));
    return evidence;
  }

  double _estimateDistanceMeters(double bboxHeightNormalized, double focalLengthPx, int frameHeightPx) {
    final bboxHeightPx = bboxHeightNormalized * frameHeightPx;
    if (bboxHeightPx <= 0) return double.infinity;
    return (_assumedPersonHeightMeters * focalLengthPx) / bboxHeightPx;
  }

  double _linearSlope(List<double> values) {
    final n = values.length;
    if (n < 2) return 0;
    final xs = List<double>.generate(n, (i) => i.toDouble());
    final meanX = xs.reduce((a, b) => a + b) / n;
    final meanY = values.reduce((a, b) => a + b) / n;
    double num = 0, den = 0;
    for (var i = 0; i < n; i++) {
      num += (xs[i] - meanX) * (values[i] - meanY);
      den += (xs[i] - meanX) * (xs[i] - meanX);
    }
    return den == 0 ? 0 : num / den;
  }

  double _headingDiff(double a, double b) {
    final diff = (a - b).abs() % 360;
    return diff > 180 ? 360 - diff : diff;
  }

  List<Evidence> _detectPossibleFollowing(
    List<TrackedObject> tracks,
    double focalLengthPx,
    int frameHeightPx,
    double fovDegrees,
    double? userHeadingDegrees,
    DateTime timestamp,
  ) {
    final out = <Evidence>[];
    for (final t in tracks.where((tr) => tr.label == 'person')) {
      if (t.boxHistory.length < 4) continue;

      final distances = t.boxHistory
          .map((box) => _estimateDistanceMeters(box[2] - box[0], focalLengthPx, frameHeightPx))
          .toList();
      final distanceSlope = _linearSlope(distances);
      final meanDistance = distances.reduce((a, b) => a + b) / distances.length;
      final trendFlatOrClosing = meanDistance > 0 && (distanceSlope / meanDistance) <= 0.05;

      final centroidXs = t.boxHistory.map((b) => (b[1] + b[3]) / 2).toList();
      final bearingRange = centroidXs.reduce((a, b) => a > b ? a : b) -
          centroidXs.reduce((a, b) => a < b ? a : b);
      final bearingStable = bearingRange < 0.15;

      bool confirmedByTurn = false;
      if (userHeadingDegrees != null) {
        final bearingOffset = (t.centroidX - 0.5) * fovDegrees;
        final objectAbsoluteBearing = (userHeadingDegrees + bearingOffset + 360) % 360;
        final history = _headingHistory.putIfAbsent(t.trackId, () => []);
        history.add(_HeadingBearingSample(timestamp, userHeadingDegrees, objectAbsoluteBearing));
        if (history.length > 10) history.removeAt(0);

        if (history.length >= 4) {
          final headingChange = _headingDiff(history.first.userHeadingDegrees, history.last.userHeadingDegrees);
          final bearingChange = _headingDiff(
              history.first.objectAbsoluteBearingDegrees, history.last.objectAbsoluteBearingDegrees);
          if (headingChange >= _turnThresholdDegrees && bearingChange <= _bearingStabilityToleranceDegrees) {
            confirmedByTurn = true;
          }
        }
      }

      if (confirmedByTurn) {
        out.add(Evidence(
          modality: Modality.vision,
          timestamp: timestamp,
          confidence: behaviorHeuristicConfidenceCap,
          riskContribution: 55 * behaviorHeuristicConfidenceCap,
          requiresCorroboration: true,
          sourceKey: 'track_${t.trackId}_possible_following',
          payload: {
            'flag': 'possible_following', 'track_id': t.trackId,
            'basis': ['heading_turn_correlation', 'distance_trend', 'bearing_stability'],
            'confirmed_by_turn': true, 'heuristic': true,
          },
        ));
      } else if (trendFlatOrClosing && bearingStable) {
        out.add(Evidence(
          modality: Modality.vision,
          timestamp: timestamp,
          confidence: behaviorHeuristicConfidenceCap * 0.7,
          riskContribution: 40 * behaviorHeuristicConfidenceCap * 0.7,
          requiresCorroboration: true,
          sourceKey: 'track_${t.trackId}_possible_following',
          payload: {
            'flag': 'possible_following', 'track_id': t.trackId,
            'basis': ['distance_trend', 'bearing_stability'],
            'confirmed_by_turn': false, 'heuristic': true,
          },
        ));
      }
    }
    return out;
  }

  List<Evidence> _detectBlockedPath(
    List<TrackedObject> tracks,
    double? userSpeedMetersPerSecond,
    DateTime timestamp,
  ) {
    final out = <Evidence>[];
    if (_recentUserSpeeds.length < 2) return out;
    final wasMoving = _recentUserSpeeds.first >= 0.8;
    final nowStationary = _recentUserSpeeds.last < _stationaryThresholdMps;
    if (!(wasMoving && nowStationary)) return out;

    for (final t in tracks) {
      final widthNormalized = t.boundingBox[3] - t.boundingBox[1];
      final centeredInFrame = t.centroidX > 0.3 && t.centroidX < 0.7;
      final largeInFrame = widthNormalized > 0.5;
      if (centeredInFrame && largeInFrame) {
        out.add(Evidence(
          modality: Modality.vision,
          timestamp: timestamp,
          confidence: behaviorHeuristicConfidenceCap,
          riskContribution: 35 * behaviorHeuristicConfidenceCap,
          requiresCorroboration: true,
          sourceKey: 'track_${t.trackId}_possible_blocked_path',
          payload: {
            'flag': 'possible_blocked_path', 'track_id': t.trackId,
            'basis': ['forward_cone_occupancy', 'bbox_width_fraction', 'speed_drop_to_zero'],
            'heuristic': true,
          },
        ));
      }
    }
    return out;
  }

  List<Evidence> _detectCloseStaticInteraction(List<TrackedObject> tracks, DateTime timestamp) {
    final out = <Evidence>[];
    final people = tracks.where((t) => t.label == 'person').toList();
    for (var i = 0; i < people.length; i++) {
      for (var j = i + 1; j < people.length; j++) {
        final a = people[i];
        final b = people[j];
        if (a.boxHistory.length < 4 || b.boxHistory.length < 4) continue;
        final overlapping = _boxesOverlap(a.boundingBox, b.boundingBox);
        final aStillness = _meanStepDisplacement(a.boxHistory);
        final bStillness = _meanStepDisplacement(b.boxHistory);
        final bothStill = aStillness < 0.02 && bStillness < 0.02;
        if (overlapping && bothStill) {
          final key = 'track_pair_${[a.trackId, b.trackId]..sort()}_close_static_interaction';
          out.add(Evidence(
            modality: Modality.vision,
            timestamp: timestamp,
            confidence: behaviorHeuristicConfidenceCap * 0.6,
            riskContribution: 25 * behaviorHeuristicConfidenceCap,
            requiresCorroboration: true,
            sourceKey: key,
            payload: {
              'flag': 'close_static_interaction', 'track_ids': [a.trackId, b.trackId],
              'basis': ['bbox_overlap', 'mutual_stillness'], 'heuristic': true,
              'note': 'Not a confrontation classifier — geometry cannot infer intent.',
            },
          ));
        }
      }
    }
    return out;
  }

  bool _boxesOverlap(List<double> a, List<double> b) {
    final yOverlap = a[0] < b[2] && a[2] > b[0];
    final xOverlap = a[1] < b[3] && a[3] > b[1];
    return yOverlap && xOverlap;
  }

  double _meanStepDisplacement(List<List<double>> history) {
    if (history.length < 2) return 0;
    double total = 0;
    for (var i = 1; i < history.length; i++) {
      final prev = history[i - 1];
      final curr = history[i];
      final dx = ((curr[1] + curr[3]) / 2) - ((prev[1] + prev[3]) / 2);
      final dy = ((curr[0] + curr[2]) / 2) - ((prev[0] + prev[2]) / 2);
      total += dx.abs() + dy.abs();
    }
    return total / (history.length - 1);
  }

  void reset() {
    _headingHistory.clear();
    _recentUserSpeeds.clear();
  }
}