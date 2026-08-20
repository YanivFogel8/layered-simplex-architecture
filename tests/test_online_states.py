"""Tests for the causal state promotion behind
scripts/bible_online_states_experiment.py."""

from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from bible_online_states_experiment import online_state_profiles  # noqa: E402


def test_every_coded_position_lands_in_exactly_one_state():
    tokens = "a b a b a c a b a b".split()
    states, _, _ = online_state_profiles(tokens, theta=2)
    total = sum(sum(c.values()) for c in states.values())
    assert total == len(tokens) - 1  # x_2 .. x_n


def test_promotion_is_causal_and_permanent():
    # theta = 2: 'a' is promoted only once it has been seen twice BEFORE
    # the position being coded.
    tokens = ["a", "x", "a", "y", "a", "z"]
    states, n_promoted, _ = online_state_profiles(tokens, theta=2)
    # positions coded: x(ctx a, seen 1) -> backoff; a(ctx x) -> backoff;
    # y(ctx a, seen 2) -> state a; a(ctx y) -> backoff; z(ctx a, seen 3) -> a
    assert n_promoted >= 1
    assert sum(states["a"].values()) == 2
    assert states["a"]["y"] == 1 and states["a"]["z"] == 1
    assert sum(states["<backoff>"].values()) == 3


def test_infinite_threshold_collapses_to_unigram():
    tokens = "a b c a b c a b c".split()
    states, n_promoted, _ = online_state_profiles(tokens, theta=10**9)
    assert n_promoted == 0
    assert set(states) == {"<backoff>"}
    assert sum(states["<backoff>"].values()) == len(tokens) - 1
