import gzip
import json
import tempfile
import unittest
from pathlib import Path

from tiger_min.data.adapters import ProcessedSequenceAdapter, RawAmazonAdapter, SequenceCorpus
from tiger_min.data.splits import build_leave_one_out_splits


class DataFlowTest(unittest.TestCase):
    def test_raw_amazon_adapter_sorts_and_remaps(self) -> None:
        rows = [
            {"reviewerID": "u1", "asin": "b", "unixReviewTime": 2},
            {"reviewerID": "u1", "asin": "a", "unixReviewTime": 1},
            {"reviewerID": "u1", "asin": "c", "unixReviewTime": 3},
            {"reviewerID": "u1", "asin": "d", "unixReviewTime": 4},
            {"reviewerID": "u1", "asin": "e", "unixReviewTime": 5},
            {"reviewerID": "u2", "asin": "x", "unixReviewTime": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Beauty_5.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            corpus = RawAmazonAdapter(path, min_sequence_length=5).load_corpus()

        self.assertEqual(corpus.num_users, 1)
        self.assertEqual(corpus.sequences[0], [0, 1, 2, 3, 4])
        self.assertEqual(corpus.item2id, {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4})

    def test_processed_adapter_accepts_dict_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequences.json"
            path.write_text(json.dumps({"u1": ["a", "b", "c", "d", "e"]}), encoding="utf-8")
            corpus = ProcessedSequenceAdapter(path, min_sequence_length=5).load_corpus()

        self.assertEqual(corpus.sequences, [[0, 1, 2, 3, 4]])
        self.assertEqual(corpus.num_items, 5)

    def test_leave_one_out_split_has_no_position_leakage(self) -> None:
        corpus = SequenceCorpus(
            sequences=[[0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]],
            user2id={"u0": 0, "u1": 1},
            item2id={str(i): i for i in range(7)},
            source="toy",
        )
        splits = build_leave_one_out_splits(corpus, max_history_length=None)

        self.assertEqual(len(splits.valid), 2)
        self.assertEqual(len(splits.test), 2)
        self.assertEqual(splits.valid[0].history, [0, 1, 2])
        self.assertEqual(splits.valid[0].target, 3)
        self.assertEqual(splits.test[0].history, [0, 1, 2, 3])
        self.assertEqual(splits.test[0].target, 4)

        user0_train = [sample for sample in splits.train if sample.user_id == 0]
        self.assertEqual([sample.target for sample in user0_train], [1, 2])
        self.assertNotIn(splits.valid[0].target, [sample.target for sample in user0_train])
        self.assertNotIn(splits.test[0].target, [sample.target for sample in user0_train])

    def test_max_history_truncates_from_left(self) -> None:
        corpus = SequenceCorpus(
            sequences=[[0, 1, 2, 3, 4, 5]],
            user2id={"u0": 0},
            item2id={str(i): i for i in range(6)},
            source="toy",
        )
        splits = build_leave_one_out_splits(corpus, max_history_length=2)

        self.assertEqual(splits.valid[0].history, [2, 3])
        self.assertEqual(splits.test[0].history, [3, 4])


if __name__ == "__main__":
    unittest.main()
