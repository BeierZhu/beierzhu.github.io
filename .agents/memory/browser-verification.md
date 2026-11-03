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

Select Chromium debugging targets by page type, never by their position in the target list.

**Why:** This environment can expose a built-in extension's background document before the actual browser page. Attaching to the first target then makes navigation checks inspect the extension rather than the application.

**How to apply:** Filter debugging targets to ordinary page documents, excluding extension URLs, before sending navigation or DOM inspection commands.
