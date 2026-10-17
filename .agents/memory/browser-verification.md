---
name: Browser verification
description: Non-obvious Chromium behavior when verifying collapsible content and isolating browser runtime files.
---

For native collapsed details, verify the open state and actual browser visibility, not a zero bounding-box height.

**Why:** Chromium can retain nonzero layout dimensions for descendants of closed details even though they are not visible.

**How to apply:** Check visibility before and after actual activation, and verify localized content in the expanded state.

Keep temporary Chromium profiles and XDG configuration/cache directories outside the Jekyll workspace.

**Why:** A temporary browser profile alone does not redirect crash-report settings; Chromium otherwise writes under the workspace configuration directory and triggers Jekyll regeneration.

**How to apply:** Explicitly direct temporary browser configuration, cache, and profile storage into temporary directories when launching browser probes.
