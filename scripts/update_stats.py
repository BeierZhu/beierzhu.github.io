#!/usr/bin/env python3
import re
import yaml
from collections import defaultdict

BIB_FILE = "_bibliography/papers.bib"
STATS_FILE = "_data/venue_stats.yml"

VENUE_GROUPS = {
    "NeurIPS": "NeurIPS/ICLR",
    "ICLR": "NeurIPS/ICLR",
    "CVPR": "CVPR/ICCV",
    "ICCV": "CVPR/ICCV",
}

EXCLUDE_VENUES = set()
OTHER_VENUES = {"FCS", "arXiv", "Thesis", "TSG"}

def parse_bib(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    entries = re.split(r"@\w+\s*\{", content)[1:]
    papers = []
    
    for entry in entries:
        abbr_match = re.search(r"abbr\s*=\s*\{([^}]+)\}", entry)
        author_match = re.search(r"author\s*=\s*\{([^}]+)\}", entry)
        
        if abbr_match and author_match:
            abbr = abbr_match.group(1).strip()
            authors = author_match.group(1).strip()
            papers.append({"abbr": abbr, "authors": authors})
    
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
    venue_order = ["NeurIPS/ICLR", "CVPR/ICCV", "AAAI", "MM", "TIP", "Others"]
    
    for venue in venue_order:
        if venue in venue_counts:
            stats.append({
                "venue": venue,
                "count": venue_counts[venue],
                "first_corresponding": venue_fc[venue]
            })
    
    for venue in sorted(venue_counts.keys()):
        if venue not in venue_order:
            stats.append({
                "venue": venue,
                "count": venue_counts[venue],
                "first_corresponding": venue_fc[venue]
            })
    
    return stats

def main():
    papers = parse_bib(BIB_FILE)
    stats = compute_stats(papers)
    
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Updated {STATS_FILE}:")
    for s in stats:
        print(f"  {s['venue']}: {s['count']} papers, {s['first_corresponding']} 1st/corresponding")
    
    total_papers = sum(s["count"] for s in stats)
    total_fc = sum(s["first_corresponding"] for s in stats)
    print(f"  Total: {total_papers} papers, {total_fc} 1st/corresponding")

if __name__ == "__main__":
    main()
