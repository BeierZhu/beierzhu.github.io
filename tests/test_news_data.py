"""Bilingual news authoring checks. Uses Jekyll's Ruby YAML parser."""

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NewsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            [
                "ruby", "-ryaml", "-rjson", "-rdate", "-e",
                """
                news = Dir.glob("_news/*.md").sort.map do |path|
                  parts = File.read(path).split(/^---\\s*$/, 3)
                  raise "Missing front matter: #{path}" unless parts.size == 3
                  metadata = YAML.safe_load(
                    parts[1], permitted_classes: [Date, Time], aliases: false
                  )
                  {path: path, metadata: metadata, body: parts[2].strip}
                end
                puts JSON.generate(news)
                """,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        cls.news = json.loads(result.stdout)

    def test_every_news_file_contains_both_languages(self):
        self.assertTrue(self.news)
        for news in self.news:
            with self.subTest(path=news["path"]):
                for field in ("content_en", "content_zh"):
                    value = news["metadata"].get(field)
                    self.assertIsInstance(value, str, field)
                    self.assertTrue(value.strip(), field)

    def test_news_metadata_is_present(self):
        for news in self.news:
            with self.subTest(path=news["path"]):
                self.assertTrue(news["metadata"].get("title"))
                self.assertTrue(news["metadata"].get("date"))
                self.assertIsInstance(news["metadata"].get("inline"), bool)

    def test_individual_news_pages_use_shared_bilingual_renderer(self):
        for news in self.news:
            with self.subTest(path=news["path"]):
                self.assertEqual(
                    news["body"], "{% include news-content.liquid item=page %}"
                )

    def test_best_paper_links_remain_on_short_labels(self):
        news = next(n for n in self.news if n["path"] == "_news/bestpaperICAIE.md")
        self.assertIn(
            "[paper](https://arxiv.org/abs/2606.25325)",
            news["metadata"]["content_en"],
        )
        self.assertIn(
            '<a href="https://arxiv.org/abs/2606.25325">论文</a>',
            news["metadata"]["content_zh"],
        )

    def test_separate_translation_file_is_no_longer_needed(self):
        self.assertFalse((ROOT / "_data/news_zh.yml").exists())


if __name__ == "__main__":
    unittest.main()
