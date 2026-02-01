#!/usr/bin/env python3
"""
Generate class statistics from papers.bib
Groups papers by supclass and class
"""

import re
import yaml
from collections import defaultdict
from pathlib import Path

BIB_FILE = Path(__file__).parent.parent / "_bibliography" / "papers.bib"
OUTPUT_FILE = Path(__file__).parent.parent / "_data" / "class_stats.yml"

SUPCLASS_ORDER = ["robust learning", "multimodal learning", "others"]
SUPCLASS_DISPLAY = {
    "robust learning": "Robust Learning",
    "multimodal learning": "Multimodal Learning",
    "others": "Others"
}

def parse_bib(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    entries = re.split(r'\n@', content)
    papers = []
    
    for entry in entries:
        if not entry.strip():
            continue
        
        supclass_match = re.search(r'supclass\s*=\s*\{([^}]+)\}', entry)
        class_match = re.search(r'class\s*=\s*\{([^}]+)\}', entry)
        
        supclass = supclass_match.group(1).strip().lower() if supclass_match else "others"
        paper_class = class_match.group(1).strip() if class_match else "Uncategorized"
        
        papers.append({
            "supclass": supclass,
            "class": paper_class
        })
    
    return papers

def compute_stats(papers):
    supclass_counts = defaultdict(int)
    class_counts = defaultdict(lambda: defaultdict(int))
    
    for paper in papers:
        supclass = paper["supclass"]
        paper_class = paper["class"]
        
        supclass_counts[supclass] += 1
        class_counts[supclass][paper_class] += 1
    
    stats = []
    for supclass in SUPCLASS_ORDER:
        if supclass not in supclass_counts:
            continue
        
        classes = []
        for class_name in sorted(class_counts[supclass].keys()):
            classes.append({
                "name": class_name,
                "count": class_counts[supclass][class_name]
            })
        
        stats.append({
            "supclass": SUPCLASS_DISPLAY.get(supclass, supclass.title()),
            "total": supclass_counts[supclass],
            "classes": classes
        })
    
    return stats

def main():
    papers = parse_bib(BIB_FILE)
    stats = compute_stats(papers)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Updated {OUTPUT_FILE}:")
    for supclass_data in stats:
        print(f"\n  {supclass_data['supclass']} ({supclass_data['total']} papers):")
        for cls in supclass_data['classes']:
            print(f"    - {cls['name']}: {cls['count']}")

if __name__ == "__main__":
    main()
