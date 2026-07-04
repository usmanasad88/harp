"""Unit tests for the pure retry/backoff policy."""

from harp.orchestrator import retry


def test_backoff_grows_exponentially():
    assert retry.backoff_seconds(1) == retry.BASE_DELAY
    assert retry.backoff_seconds(2) == retry.BASE_DELAY * 2
    assert retry.backoff_seconds(3) == retry.BASE_DELAY * 4


def test_backoff_is_capped():
    assert retry.backoff_seconds(50) == retry.MAX_DELAY


def test_gives_up_after_max_attempts():
    assert not retry.should_give_up(retry.MAX_ATTEMPTS - 1, 0.0)
    assert retry.should_give_up(retry.MAX_ATTEMPTS, 0.0)


def test_gives_up_after_max_elapsed():
    assert not retry.should_give_up(1, retry.MAX_ELAPSED - 1)
    assert retry.should_give_up(1, retry.MAX_ELAPSED)
