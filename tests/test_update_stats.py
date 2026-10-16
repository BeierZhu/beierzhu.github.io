import unittest

from scripts.update_stats import compute_stats, compute_summary


class VenueStatsTests(unittest.TestCase):
    def paper(self, venue, authors="Zhu, Beier", award=None):
        return {"abbr": venue, "authors": authors, "award": award}

    def test_preprint_is_only_listed_under_others(self):
        self.assertEqual(
            compute_stats([self.paper("Preprint")]),
            [{"venue": "Others", "count": 1, "first_corresponding": 1}],
        )

    def test_preprints_merge_with_existing_others_counts(self):
        papers = [
            self.paper("Preprint"),
            self.paper("arXiv", "Doe, Jane and Zhu^, Beier"),
            self.paper("FCS", "Doe, Jane"),
        ]
        self.assertEqual(
            compute_stats(papers),
            [{"venue": "Others", "count": 3, "first_corresponding": 2}],
        )

    def test_regular_venues_and_overall_totals_are_preserved(self):
        papers = [
            self.paper("NeurIPS", award="Oral"),
            self.paper("Preprint", "Doe, Jane"),
        ]
        self.assertEqual(
            compute_stats(papers),
            [
                {"venue": "NeurIPS", "count": 1, "first_corresponding": 1},
                {"venue": "Others", "count": 1, "first_corresponding": 0},
            ],
        )
        self.assertEqual(
            compute_summary(papers),
            {"total": 2, "first_corresponding": 1, "distinguished": 1},
        )


if __name__ == "__main__":
    unittest.main()
