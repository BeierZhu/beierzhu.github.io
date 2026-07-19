#!/usr/bin/env python3
import re
import yaml
from collections import defaultdict

BIB_FILE = "_bibliography/papers.bib"
STATS_FILE = "_data/venue_stats.yml"
SUMMARY_FILE = "_data/summary_stats.yml"

VENUE_GROUPS = {
    # "NeurIPS": "NeurIPS/ICLR/ICML",
    # "ICLR": "NeurIPS/ICLR/ICML",
    # "ICML": "NeurIPS/ICLR/ICML",
    # "CVPR": "CVPR/ICCV",
    # "ICCV": "CVPR/ICCV",
}

EXCLUDE_VENUES = set()
OTHER_VENUES = {"FCS", "arXiv", "Thesis", "TSG"}

AWARD_TYPES = {"Oral", "Spotlight", "Highlight"}


def parse_bib(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    entries = re.split(r"@\w+\s*\{", content)[1:]
    papers = []

    for entry in entries:
        abbr_match = re.search(r"abbr\s*=\s*\{([^}]+)\}", entry)
        author_match = re.search(r"author\s*=\s*\{([^}]+)\}", entry)
        award_match = re.search(r"award\s*=\s*\{([^}]+)\}", entry)

        if abbr_match and author_match:
            abbr = abbr_match.group(1).strip()
            authors = author_match.group(1).strip()
            award = award_match.group(1).strip() if award_match else None
            papers.append({"abbr": abbr, "authors": authors, "award": award})

    return papers


def is_first_or_corresponding(authors):
    author_list = [a.strip() for a in authors.split(" and ")]
    if not author_list:
        return False

    first_author = author_list[0]

    if "Zhu" in first_author and "Beier" in first_author:
        return True

    if "*" in first_author:
        for author in author_list:
            if ("Zhu" in author and "Beier" in author) and "*" in author:
                return True

    for author in author_list:
        if ("Zhu" in author and "Beier" in author) and "^" in author:
            return True

    return False


def compute_stats(papers):
    venue_counts = defaultdict(int)
    venue_fc = defaultdict(int)

    for paper in papers:
        abbr = paper["abbr"]

        if abbr in OTHER_VENUES:
            venue = "Others"
        else:
            venue = VENUE_GROUPS.get(abbr, abbr)

        venue_counts[venue] += 1

        if is_first_or_corresponding(paper["authors"]):
            venue_fc[venue] += 1

    stats = []
    venue_order = [
        "NeurIPS",
        "ICLR",
        "ICML",
        "CVPR",
        "ICCV",
        "AAAI",
        "MM",
        "ACL",
        "TPAMI",
        "TIP",
        "Others",
    ]

    for venue in venue_order:
        if venue in venue_counts:
            stats.append(
                {
                    "venue": venue,
                    "count": venue_counts[venue],
                    "first_corresponding": venue_fc[venue],
                }
            )

    for venue in sorted(venue_counts.keys()):
        if venue not in venue_order:
            stats.append(
                {
                    "venue": venue,
                    "count": venue_counts[venue],
                    "first_corresponding": venue_fc[venue],
                }
            )

    return stats


def compute_summary(papers):
    total = len(papers)
    first_corresponding = sum(1 for p in papers if is_first_or_corresponding(p["authors"]))

    award_counts = defaultdict(int)
    for p in papers:
        if p["award"] and p["award"] in AWARD_TYPES:
            award_counts[p["award"]] += 1

    return {
        "total": total,
        "first_corresponding": first_corresponding,
        "oral": award_counts.get("Oral", 0),
        "spotlight": award_counts.get("Spotlight", 0),
        "highlight": award_counts.get("Highlight", 0),
    }


def main():
    papers = parse_bib(BIB_FILE)
    stats = compute_stats(papers)
    summary = compute_summary(papers)

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated {STATS_FILE}:")
    for s in stats:
        print(
            f"  {s['venue']}: {s['count']} papers, {s['first_corresponding']} 1st/corresponding"
        )

    print(f"\nUpdated {SUMMARY_FILE}:")
    print(f"  Total papers:            {summary['total']}")
    print(f"  First/corresponding:     {summary['first_corresponding']}")
    print(f"  Oral:                    {summary['oral']}")
    print(f"  Spotlight:               {summary['spotlight']}")
    print(f"  Highlight:               {summary['highlight']}")


if __name__ == "__main__":
    main()
