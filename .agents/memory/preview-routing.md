---
name: Preview routing checks
description: Distinguish app rendering from the URL and port selected by the Preview panel.
---

Check both the locally rendered application and the proxied development URL when diagnosing a blank Preview. Do not infer the user's Preview panel is fixed from a local screenshot alone.

**Why:** This project's homepage rendered locally while an obsolete exposed-port mapping made the default development URL return 502. Preview can also retain a previously selected port.

**How to apply:** Verify active workflow ports and the exposed mappings before changing them. Keep only needed service mappings, test the default development URL and its assets, and distinguish those checks from visibility of the user's own panel.
