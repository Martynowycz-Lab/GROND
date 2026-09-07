from grond.runtime import auto_chunksize, auto_worker_count


def test_auto_worker_count_honors_explicit():
    assert auto_worker_count(7, 0.85) == 7


def test_auto_worker_count_uses_fraction(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 128)
    assert auto_worker_count(0, 0.85) == 108


def test_auto_chunksize_scales_with_workers():
    assert auto_chunksize(0, 8) == 4
    assert auto_chunksize(0, 32) == 2
    assert auto_chunksize(0, 128) == 1
    assert auto_chunksize(5, 128) == 5
