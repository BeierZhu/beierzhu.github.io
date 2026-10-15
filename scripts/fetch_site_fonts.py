#!/usr/bin/env python3
"""Fetch checksum-pinned fonts before Jekyll builds the self-hosted website.

Browsers use assets/fonts/site-fonts.css and same-origin WOFF2 files only.
Keep the URLs and SHA-256 values pinned when updating the font versions.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "fonts"
ASSETS = json.loads(r'''[
  [
    "material-icons-v145-normal-fallback.woff2",
    "https://fonts.gstatic.com/s/materialicons/v145/flUhRq6tzZclQEJ-Vdg-IuiaDsNcIhQ8tQ.woff2",
    "4a4dbc62fa335e411b94a532be091c58c0c0c4fa731339f11722577d3cf6443b"
  ],
  [
    "roboto-v51-italic-cyrillic-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkC3kaSTbQWt4N.woff2",
    "9d15310cc10cbf87b1a68bcd4a423eaa9a9cbafca704618b8b53ec03db0cb142"
  ],
  [
    "roboto-v51-italic-cyrillic.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkAnkaSTbQWt4N.woff2",
    "8e05333d3e339a143a86b7382b08611116e96275ffb19efb09dc6af0fd7bda39"
  ],
  [
    "roboto-v51-italic-greek-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkCnkaSTbQWt4N.woff2",
    "6417a7ddc238c2728b4ef2984073585c1e3a8572e8d78c4d06ee67a3dc806569"
  ],
  [
    "roboto-v51-italic-greek.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkBXkaSTbQWt4N.woff2",
    "4fab70a79ac72b42fd19547b58351a9f69d96a95caa1f6e86b14c15f5ce2192f"
  ],
  [
    "roboto-v51-italic-math.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkenkaSTbQWt4N.woff2",
    "004fc38a79198136b08684bea9c79019789e059e663fd946e6fa6b468c409b5c"
  ],
  [
    "roboto-v51-italic-symbols.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkaHkaSTbQWt4N.woff2",
    "0b4720fbd73fe37b8566108a91acc25b3a35dbcaeefa14467cdbc23fe90c9c7e"
  ],
  [
    "roboto-v51-italic-vietnamese.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkCXkaSTbQWt4N.woff2",
    "ab2869d94d84f4911d54fb647180f42ebd04929d35a910da3eec18a1b4c5169a"
  ],
  [
    "roboto-v51-italic-latin-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkCHkaSTbQWt4N.woff2",
    "e9f07da0193462efe80d281f293260116439d265ec1ce4c49d921378d97e2ac6"
  ],
  [
    "roboto-v51-italic-latin.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO5CnqEu92Fr1Mu53ZEC9_Vu3r1gIhOszmkBnkaSTbQWg.woff2",
    "9f06741ff731c4555398df16ce523ee8fd24dfd9dfd2ff2a6c9d63239cf0e9f7"
  ],
  [
    "roboto-v51-normal-cyrillic-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3GUBHMdazTgWw.woff2",
    "e44224eb71a96cb33bd046ded16a4e22bb441b170bdfbcba0268a754141b97c8"
  ],
  [
    "roboto-v51-normal-cyrillic.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3iUBHMdazTgWw.woff2",
    "32886c5ec61a0d06e5677a98c5ed78a5f1c788c18c5e8d7dcd931c4b6dc19830"
  ],
  [
    "roboto-v51-normal-greek-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3CUBHMdazTgWw.woff2",
    "5fdfeaf926cac990770dfa66011bdb87d6b69f43c42aa759dcdcf0a7a197c19a"
  ],
  [
    "roboto-v51-normal-greek.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3-UBHMdazTgWw.woff2",
    "4813e3dfbbb23708c817ef7fd3fd9c82cd0d33c778def0897f0447f1c364ce64"
  ],
  [
    "roboto-v51-normal-math.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMawCUBHMdazTgWw.woff2",
    "297193369eea600161f4b5219bd7b08850180fcfe0752f1a2eaa436575cd03b7"
  ],
  [
    "roboto-v51-normal-symbols.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMaxKUBHMdazTgWw.woff2",
    "fab8a0056d92fdd8d64fed4b049744144e1d24d08762c513bff78265299ed7a3"
  ],
  [
    "roboto-v51-normal-vietnamese.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3OUBHMdazTgWw.woff2",
    "2071148c26bd0430cad316373c26db399e0cf8f6275b9e1a2c1b831f05ee7dc3"
  ],
  [
    "roboto-v51-normal-latin-ext.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3KUBHMdazTgWw.woff2",
    "cdd53dc5576a79f35cf280dfbd9c1cb2a5dd2f0ede30afba9b9ccdc7fe12e7d0"
  ],
  [
    "roboto-v51-normal-latin.woff2",
    "https://fonts.gstatic.com/s/roboto/v51/KFO7CnqEu92Fr1ME7kSn66aGLdTylUAMa3yUBHMdazQ.woff2",
    "0a44e0bb6ba5c8537e8814c148ef7755f1bce12112361231f595ecc584a18d7a"
  ],
  [
    "roboto-slab-v36-normal-cyrillic-ext.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufA5qWr4xCCQ_k.woff2",
    "7eea44b5371f1f84d39bad778eed8b9c179bd391fe70177fa13a559024217b6d"
  ],
  [
    "roboto-slab-v36-normal-cyrillic.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufJ5qWr4xCCQ_k.woff2",
    "07a6f958db51de42694ef87e2c01b9cf5ee415047b2fd17ab1c313c772a6804e"
  ],
  [
    "roboto-slab-v36-normal-greek-ext.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufB5qWr4xCCQ_k.woff2",
    "f32dd3fe9582bebb93efd811681c670955c3ac6780cdd1adea1ad9ca581d9763"
  ],
  [
    "roboto-slab-v36-normal-greek.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufO5qWr4xCCQ_k.woff2",
    "91d28703bda49c4bfc4072b8de79af647fd385ee6ba5782ea4c4784512454697"
  ],
  [
    "roboto-slab-v36-normal-vietnamese.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufC5qWr4xCCQ_k.woff2",
    "e77f0a7a71bf848476fd85c91e9ca1b93f6367480150fb0179662a41436dbdb2"
  ],
  [
    "roboto-slab-v36-normal-latin-ext.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufD5qWr4xCCQ_k.woff2",
    "3c81cc23c315d97e91d07fa69b006cba1267592d0822165e6045dde0eb4e125e"
  ],
  [
    "roboto-slab-v36-normal-latin.woff2",
    "https://fonts.gstatic.com/s/robotoslab/v36/BngMUXZYTXPIvIBgJJSb6ufN5qWr4xCC.woff2",
    "317b2dafcfcd18ae868e7cb3c5a33a323999bc4c3c400e237801efc2a3a74aac"
  ],
  [
    "LICENSE-Roboto.txt",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/OFL.txt",
    "061402327a96aadb0bfb694a960ed289ecd38d383e396243831ab81feb109c41"
  ],
  [
    "LICENSE-Roboto-Slab.txt",
    "https://raw.githubusercontent.com/googlefonts/robotoslab/main/LICENSE.txt",
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
  ],
  [
    "LICENSE-Material-Icons.txt",
    "https://raw.githubusercontent.com/google/material-design-icons/master/LICENSE",
    "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd"
  ]
]''')


def fetch_asset(record):
    name, url, expected_sha256 = record
    target = DEST / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        return
    fd, temporary = tempfile.mkstemp(prefix=".font-", dir=DEST)
    os.close(fd)
    try:
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--retry", "2", "--connect-timeout", "10", "--max-time", "60",
             url, "--output", temporary],
            check=True,
        )
        actual = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch_asset, ASSETS))
    sources = [
        "Self-hosted website fonts", "",
        "Roboto v51: normal and italic, weights 100-900 (SIL OFL 1.1).",
        "Roboto Slab v36: normal, weights 100-900 (Apache 2.0).",
        "Material Icons v145: normal, weight 400 (Apache 2.0).", "",
        "Font files are unmodified Google Fonts downloads. Each asset is checked",
        "against its pinned SHA-256 before being included in the site build.",
        "The accompanying LICENSE-*.txt files contain the full licenses.", "",
    ]
    for name, url, digest in ASSETS:
        sources.extend([name, f"  Source: {url}", f"  SHA-256: {digest}", ""])
    (DEST / "FONT-SOURCES.txt").write_text("\n".join(sources), encoding="utf-8")
    print(f"Verified {len(ASSETS)} local font and license assets.")


if __name__ == "__main__":
    main()
