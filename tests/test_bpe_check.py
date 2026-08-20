"""Tests for the BPE trainer/encoder behind scripts/bible_bpe_check.py."""

from collections import Counter
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from bible_bpe_check import END, encode, train_bpe  # noqa: E402


CORPUS = ("the cat sat on the mat " * 20 + "the bat and the rat ").split()


def test_training_is_deterministic():
    words = Counter(CORPUS)
    assert train_bpe(words, 40) == train_bpe(words, 40)


def test_vocab_size_is_respected():
    words = Counter(CORPUS)
    merges = train_bpe(words, 30)
    stream = encode(CORPUS, merges)
    vocab = set(stream)
    # base symbols + merges never exceed the target
    base = {ch for w in words for ch in w} | {w[-1] + END for w in words}
    assert len({s for s in vocab}) <= 30 or len(base) >= 30


def test_encoding_is_lossless():
    words = Counter(CORPUS)
    merges = train_bpe(words, 35)
    stream = encode(CORPUS, merges)
    # remove end-of-word markers and re-split: must reproduce the words
    text = "".join(stream)
    rebuilt = text.replace(END, " ").split()
    assert rebuilt == CORPUS


def test_frequent_word_becomes_single_token():
    words = Counter(CORPUS)
    merges = train_bpe(words, 60)
    stream = encode(["the"], merges)
    assert stream == ["the" + END]
