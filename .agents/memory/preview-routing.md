---
name: Preview routing checks
description: Distinguish app rendering from the URL and port selected by the Preview panel.
---

Check both the locally rendered application and the proxied development URL when diagnosing a blank Preview. Do not infer the user's Preview panel is fixed from a local screenshot alone.

**Why:** This project's homepage rendered locally while an obsolete exposed-port mapping made the default development URL return 502. Preview can also retain a previously selected port.

**How to apply:** Verify active workflow ports and the exposed mappings before changing them. Keep only needed service mappings, test the default development URL and its assets, and distinguish those checks from visibility of the user's own panel.

## Workspace controls

Use the user's actual screenshot to identify available workspace controls. An active workflow does not prove that a Run or Stop button is visible. Design/Build mode tabs are not rebuild commands.

**Why:** The user's workspace screenshot showed no Run/Stop control despite a running workflow. Repeated directions based solely on generic documentation did not match their interface.

**How to apply:** Do not invent a button location. If the screenshot shows Shell but no Run/Stop control, provide the verified project build command as an alternative.

## Verify page content, not only status

An HTTP 200 can be a WEBrick directory listing rather than the requested page.

**Why:** A standalone Liquid source page retained its extension in the generated index, so the server returned a directory listing despite a successful build and HTTP 200.

**How to apply:** Use HTML or Markdown source extensions for standalone pages; reserve Liquid extensions for layouts/includes. Check the generated HTML index and expected rendered content before declaring a route working.
