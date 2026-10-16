import tempfile
import unittest
from itertools import permutations
from pathlib import Path

from scripts.update_class_stats import compute_stats, parse_bib


class ClassStatsTests(unittest.TestCase):
    def parse_fields(self, fields):
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "papers.bib"
            filepath.write_text(
                "@article{example,\n" + fields + "\n}\n", encoding="utf-8"
            )
            return parse_bib(filepath)

    def test_class_before_supclass(self):
        self.assertEqual(
            self.parse_fields(
                "class = {Reasoning},\nsupclass = {LLMs & VLMs},"
            ),
            [{"supclass": "llms & vlms", "class": "Reasoning"}],
        )

    def test_supclass_before_class(self):
        self.assertEqual(
            self.parse_fields(
                "supclass = {LLMs & VLMs},\nclass = {Reasoning},"
            ),
            [{"supclass": "llms & vlms", "class": "Reasoning"}],
        )

    def test_missing_class_is_not_read_from_supclass(self):
        self.assertEqual(
            self.parse_fields("supclass = {Diffusion Models},"),
            [{"supclass": "diffusion models", "class": "Uncategorized"}],
        )

    def test_missing_supclass_keeps_default(self):
        self.assertEqual(
            self.parse_fields("class = {Reasoning},"),
            [{"supclass": "others", "class": "Reasoning"}],
        )

    def test_prefixed_fields_are_not_classification_fields(self):
        self.assertEqual(
            self.parse_fields("subclass = {Reasoning},\nnotsupclass = {LLMs & VLMs},"),
            [{"supclass": "others", "class": "Uncategorized"}],
        )

    def test_statistics_are_identical_for_all_field_orders(self):
        fields = [
            "title = {Example}",
            "class = {Reasoning}",
            "supclass = { LLMs & VLMs }",
        ]
        expected = [
            {
                "supclass": "LLMs & VLMs",
                "total": 1,
                "classes": [{"name": "Reasoning", "count": 1}],
            }
        ]
        for order in permutations(fields):
            for separator in (",\n", ", "):
                with self.subTest(order=order, separator=separator):
                    papers = self.parse_fields(separator.join(order) + ",")
                    self.assertEqual(compute_stats(papers), expected)


if __name__ == "__main__":
    unittest.main()
