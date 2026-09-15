"""Counter-based profiling must exclude startup and reject missing progress."""
from hpc_runs.hosts.g500.profile_training import measured_fps


def test_startup_does_not_reduce_steady_throughput():
    samples = [(t, 0) for t in range(0, 100, 5)]
    samples += [(t, (t - 90) * 100) for t in range(100, 200, 5)]
    assert measured_fps(samples) == 100


def test_missing_or_insufficient_progress_is_unavailable():
    assert measured_fps([(0, 0), (100, 0)]) is None
    assert measured_fps([(0, 10), (10, 20)]) is None


def test_stalled_frames_are_not_reported_as_positive_throughput():
    assert measured_fps([(t, 100) for t in range(0, 100, 5)]) == 0


def test_fast_trial_uses_distinct_completed_batches_when_warmup_consumes_run():
    samples = [(0, 0), (5, 100), (10, 100), (15, 200), (20, 200), (30, 400)]
    assert measured_fps(samples) == 12
