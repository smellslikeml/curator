"""Unit tests for the active-curation block.

These exercise the uncertainty x diversity selection, dual-annotator consistency
check and budget-eviction policies with a deterministic fake annotator (no API
calls). They also import and assert against pre-existing curator modules --
``curator.LLM`` and the sibling ``Raft`` block -- so the new block is verified
against the established block API rather than only self-tested.
"""

from collections import defaultdict

from datasets import Dataset

from bespokelabs import curator
from bespokelabs.curator.blocks.active_curation import ActiveCuration, PoolAnnotator, labels_agree, select_batch
from bespokelabs.curator.blocks.raft import Raft


class _FakeAnnotator:
    """Deterministic annotator implementing the `Annotator` protocol.

    Each successive annotation of a given id pulls the next label from a per-id
    schedule, so the scoring pass and the consistency pass are fully controlled.
    """

    def __init__(self, labels_by_vote, confidence=None):
        """Store per-id label schedules and confidences.

        Args:
            labels_by_vote: ``{id: [label_vote_1, label_vote_2, ...]}``. An id
                absent from the map always labels as ``"label_<id>"``.
            confidence: ``{id: float}`` confidence per id (default 0.5).
        """
        self.labels_by_vote = labels_by_vote
        self.confidence = confidence or {}
        self.counts = defaultdict(int)

    def annotate(self, items, working_dir=None):
        """Return one annotated record per input item, advancing per-id votes."""
        out = []
        for item in items:
            rid = item["id"]
            vote = self.counts[rid]
            self.counts[rid] += 1
            schedule = self.labels_by_vote.get(rid)
            if schedule is None:
                label = f"label_{rid}"
            else:
                label = schedule[vote] if vote < len(schedule) else schedule[-1]
            out.append(
                {
                    "id": rid,
                    "text": item.get("text", ""),
                    "label": label,
                    "confidence": self.confidence.get(rid, 0.5),
                    "reasoning": "",
                }
            )
        return out


def test_labels_agree_normalizes():
    """Cosmetic label differences should not fail the consistency check."""
    assert labels_agree({"label": " Cat "}, {"label": "cat"}) is True
    assert labels_agree({"label": "YES"}, {"label": " yes "}) is True
    assert labels_agree({"label": "a"}, {"label": "b"}) is False


def test_select_batch_picks_most_uncertain_first():
    """With batch size 1, the most uncertain item is selected."""
    scored = [
        {"id": "a", "text": "some text", "confidence": 0.9},
        {"id": "b", "text": "other text", "confidence": 0.2},
        {"id": "c", "text": "more text", "confidence": 0.5},
    ]
    batch = select_batch(scored, k=1)
    assert [r["id"] for r in batch] == ["b"]


def test_select_batch_avoids_near_duplicates():
    """The diversity term should skip a high-uncertainty duplicate of a pick."""
    scored = [
        {"id": "x", "text": "alpha alpha alpha", "confidence": 0.1},
        {"id": "x2", "text": "alpha alpha alpha", "confidence": 0.1},
        {"id": "y", "text": "zeta zeta zeta", "confidence": 0.2},
    ]
    batch = select_batch(scored, k=2)
    ids = [r["id"] for r in batch]
    assert ids[0] == "x"
    # "x2" is a near-duplicate of "x" (zero marginal diversity) so "y" is preferred.
    assert "x2" not in ids
    assert ids[1] == "y"


def test_select_batch_empty_and_zero_k():
    """Edge cases: empty input or k<=0 yield an empty batch."""
    assert select_batch([], k=3) == []
    assert select_batch([{"id": "a", "text": "t", "confidence": 0.5}], k=0) == []


def test_consistency_only_promotes_agreed_labels():
    """A label is promoted only when the two annotator passes agree."""
    annotator = _FakeAnnotator(
        labels_by_vote={"a": ["A", "A"], "b": ["B", "X"], "c": ["C", "C"]},
        confidence={"a": 0.5, "b": 0.5, "c": 0.5},
    )
    block = ActiveCuration(model="fake", batch_size=3, annotator_budget=6, consistency=True, annotator=annotator)
    out = block([{"id": "a", "text": "aaa"}, {"id": "b", "text": "bbb"}, {"id": "c", "text": "ccc"}])

    assert isinstance(out, Dataset)
    labels = {row["id"]: row["label"] for row in out}
    assert labels == {"a": "A", "c": "C"}  # "b" disagreed -> deferred then evicted
    assert block.last_stats["labeled"] == 2
    assert block.last_stats["evicted"] == 1
    assert block.last_stats["spent"] <= block.annotator_budget


def test_consistency_disabled_promotes_without_second_vote():
    """With consistency off, every selected item is promoted (no second pass)."""
    annotator = _FakeAnnotator(labels_by_vote={}, confidence={"a": 0.5, "b": 0.5, "c": 0.5})
    block = ActiveCuration(model="fake", batch_size=3, annotator_budget=10, consistency=False, annotator=annotator)
    out = block([{"id": "a", "text": "aaa"}, {"id": "b", "text": "bbb"}, {"id": "c", "text": "ccc"}])

    assert sorted(out["id"]) == ["a", "b", "c"]
    assert block.last_stats["labeled"] == 3
    assert block.last_stats["evicted"] == 0
    # No consistency pass, so only the single scoring pass was spent.
    assert block.last_stats["spent"] == 3


def test_budget_triggers_batch_eviction():
    """A budget too small to reach the consistency pass evicts the pool."""
    pool = [{"id": str(i), "text": f"item number {i}"} for i in range(5)]
    annotator = _FakeAnnotator(labels_by_vote={}, confidence={r["id"]: 0.5 for r in pool})
    block = ActiveCuration(model="fake", batch_size=2, annotator_budget=4, consistency=True, annotator=annotator)
    out = block(pool)

    assert len(out) == 0  # nothing could be promoted before the budget ran out
    assert block.last_stats["evicted"] == 5  # the entire pool was evicted
    assert block.last_stats["spent"] == 4
    assert block.last_stats["spent"] <= block.annotator_budget


def test_synthesizes_missing_ids_from_text():
    """Rows without an explicit id should still round-trip through the loop."""
    annotator = _FakeAnnotator(labels_by_vote={}, confidence={"_": 0.5})
    block = ActiveCuration(model="fake", batch_size=2, annotator_budget=6, consistency=False, annotator=annotator)
    out = block([{"text": "hello"}, {"text": "world"}])

    ids = sorted(out["id"])
    assert len(ids) == 2
    assert all(isinstance(i, str) and i for i in ids)


def test_pool_annotator_is_a_curator_llm():
    """The default annotator subclasses the pre-existing ``curator.LLM``."""
    assert issubclass(PoolAnnotator, curator.LLM)


def test_block_sits_alongside_existing_blocks():
    """The new block shares the blocks package and call convention with ``Raft``."""
    import bespokelabs.curator.blocks as blocks_pkg

    assert Raft.__module__.startswith(str(blocks_pkg.__name__))
    assert ActiveCuration.__module__.startswith(str(blocks_pkg.__name__))
    assert callable(Raft) and callable(ActiveCuration)
