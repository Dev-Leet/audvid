class TrackedObject {
  final int trackId;
  final String label;
  List<double> boundingBox;
  final List<List<double>> boxHistory;
  int framesSinceSeen;

  TrackedObject({
    required this.trackId,
    required this.label,
    required this.boundingBox,
    List<List<double>>? boxHistory,
    this.framesSinceSeen = 0,
  }) : boxHistory = boxHistory ?? [boundingBox];

  double get centroidY => (boundingBox[0] + boundingBox[2]) / 2;
  double get centroidX => (boundingBox[1] + boundingBox[3]) / 2;
  double get heightNormalized => boundingBox[2] - boundingBox[0];

  /// AUDIT FIX (T-13): deep copy for safe hand-off to UI/stream consumers.
  TrackedObject _snapshot() => TrackedObject(
        trackId: trackId,
        label: label,
        boundingBox: List<double>.from(boundingBox),
        boxHistory: boxHistory.map((b) => List<double>.from(b)).toList(),
        framesSinceSeen: framesSinceSeen,
      );
}

class ObjectTracker {
  final Map<int, TrackedObject> _tracks = {};
  int _nextId = 0;
  final int maxMissedFrames;
  final double iouMatchThreshold;
  final int maxHistoryPerTrack;

  ObjectTracker({
    this.maxMissedFrames = 5,
    this.iouMatchThreshold = 0.3,
    this.maxHistoryPerTrack = 10,
  });

  List<TrackedObject> update(List<({String label, List<double> box})> detections) {
    final unmatchedDetections = List.of(detections);
    final matchedTrackIds = <int>{};

    for (final entry in _tracks.entries) {
      final track = entry.value;
      ({String label, List<double> box})? best;
      double bestIou = 0;

      for (final det in unmatchedDetections) {
        if (det.label != track.label) continue;
        final iou = _iou(track.boundingBox, det.box);
        if (iou > bestIou) {
          bestIou = iou;
          best = det;
        }
      }

      if (best != null && bestIou >= iouMatchThreshold) {
        track.boundingBox = best.box;
        track.boxHistory.add(best.box);
        if (track.boxHistory.length > maxHistoryPerTrack) track.boxHistory.removeAt(0);
        track.framesSinceSeen = 0;
        matchedTrackIds.add(track.trackId);
        unmatchedDetections.remove(best);
      } else {
        track.framesSinceSeen++;
      }
    }

    for (final det in unmatchedDetections) {
      final id = _nextId++;
      _tracks[id] = TrackedObject(trackId: id, label: det.label, boundingBox: det.box);
    }

    _tracks.removeWhere((id, t) => t.framesSinceSeen > maxMissedFrames);

    // AUDIT FIX (T-13): return immutable SNAPSHOTS instead of live,
    // in-place-mutated TrackedObject references. Without this, a UI
    // consumer whose repaint lags behind the capture loop could paint a
    // track's already-advanced state rather than the state at the moment
    // it was streamed — undermining the app's core purpose of showing
    // trustworthy visual feedback. Downstream evidence/heuristic
    // computation only reads these values, so snapshotting has no effect
    // on correctness there.
    return _tracks.values.map((t) => t._snapshot()).toList();
  }

  double _iou(List<double> a, List<double> b) {
    final yMin = a[0] > b[0] ? a[0] : b[0];
    final xMin = a[1] > b[1] ? a[1] : b[1];
    final yMax = a[2] < b[2] ? a[2] : b[2];
    final xMax = a[3] < b[3] ? a[3] : b[3];
    final interH = (yMax - yMin).clamp(0, double.infinity);
    final interW = (xMax - xMin).clamp(0, double.infinity);
    final interArea = interH * interW;
    final areaA = (a[2] - a[0]) * (a[3] - a[1]);
    final areaB = (b[2] - b[0]) * (b[3] - b[1]);
    final union = areaA + areaB - interArea;
    return union <= 0 ? 0 : interArea / union;
  }

  void reset() {
    _tracks.clear();
    _nextId = 0;
  }
}