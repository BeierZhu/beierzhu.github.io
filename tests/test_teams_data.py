import json
import re
import unittest
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


class TeamsDataTests(unittest.TestCase):
    def setUp(self):
        self.files = sorted((ROOT / "_data" / "teams").glob("*.json"))
        self.assertTrue(self.files, "At least one student JSON record is required.")

    def test_each_student_has_valid_bilingual_details(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                self.assertIn(student["degree"], ("phd", "master"))
                for language in ("en", "zh"):
                    self.assertIsInstance(student["name"][language], str)
                    self.assertTrue(student["name"][language].strip())
                self.assertRegex(student["enrollment"], r"^\d{4}-\d{2}$")
                datetime.strptime(student["enrollment"], "%Y-%m")
                self.assertTrue(
                    re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", student["email"])
                )

    def test_each_student_has_bilingual_research(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                if student.get("graduated") is True:
                    continue
                for language in ("en", "zh"):
                    self.assertIsInstance(student["research"][language], str)
                    self.assertTrue(student["research"][language].strip())

    def test_graduation_status_is_a_boolean_when_provided(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                if "graduated" in student:
                    self.assertIsInstance(student["graduated"], bool)

    def test_graduated_students_have_a_year_and_bilingual_destination(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                if student.get("graduated") is not True:
                    continue
                year = student["graduation_year"]
                self.assertIs(type(year), int)
                self.assertTrue(1000 <= year <= 9999)
                self.assertGreaterEqual(year, int(student["enrollment"][:4]))
                for language in ("en", "zh"):
                    destination = student["graduation_destination"][language]
                    self.assertIsInstance(destination, str)
                    self.assertTrue(destination.strip())

    def test_photos_are_existing_local_assets(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                self.assertTrue(student["photo"].startswith("/assets/"))
                photo = (ROOT / student["photo"].lstrip("/")).resolve()
                self.assertTrue(photo.is_relative_to(ROOT))
                self.assertTrue(photo.is_file(), str(photo))

    def test_optional_homepage_accepts_only_absolute_web_urls(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                homepage = student.get("homepage", "")
                self.assertIsInstance(homepage, str)
                if homepage.strip():
                    url = urlparse(homepage)
                    self.assertIn(url.scheme, ("http", "https"))
                    self.assertTrue(url.netloc)

    def test_optional_google_scholar_accepts_only_absolute_web_urls(self):
        for filepath in self.files:
            with self.subTest(student=filepath.name):
                student = json.loads(filepath.read_text(encoding="utf-8"))
                scholar = student.get("google_scholar", "")
                self.assertIsInstance(scholar, str)
                if scholar.strip():
                    url = urlparse(scholar)
                    self.assertIn(url.scheme, ("http", "https"))
                    self.assertTrue(url.netloc)


if __name__ == "__main__":
    unittest.main()
