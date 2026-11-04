---
name: Browser verification
description: Non-obvious Chromium behavior when verifying collapsible content and isolating browser runtime files.
---

For native collapsed details, verify the open state and actual browser visibility, not a zero bounding-box height.

**Why:** Chromium can retain nonzero layout dimensions for descendants of closed details even though they are not visible.

**How to apply:** Check visibility before and after actual activation, and verify localized content in the expanded state.

For collapsible charts, compare the rendered plot and label geometry over repeated open/close cycles, not only the outer container dimensions or chart-node count.

**Why:** An earlier check confirmed the container width and absence of duplicate chart nodes, but missed a visibly compressed chart on reopening.

**How to apply:** Compare SVG dimensions, bar bounds, and label font sizes on the first and subsequent opens at both desktop and mobile widths.

For initially hidden MathJax examples, wait for typesetting and the fonts requested by the first visible render before comparing formula geometry.

**Why:** Opening hidden math can initiate font loading after the page load event. Measuring immediately can make normal font substitution look like a show/hide layout regression.

**How to apply:** Await MathJax startup, reveal the example, allow a rendering frame, await document.fonts.ready, and allow another frame before comparing formula bounds.

Keep temporary Chromium profiles and XDG configuration/cache directories outside the Jekyll workspace.

**Why:** A temporary browser profile alone does not redirect crash-report settings; Chromium otherwise writes under the workspace configuration directory and triggers Jekyll regeneration.

**How to apply:** Explicitly direct temporary browser configuration, cache, and profile storage into temporary directories when launching browser probes.

Select Chromium debugging targets by page type, never by their position in the target list.

**Why:** This environment can expose a built-in extension's background document before the actual browser page. Attaching to the first target then makes navigation checks inspect the extension rather than the application.

**How to apply:** Filter debugging targets to ordinary page documents, excluding extension URLs, before sending navigation or DOM inspection commands.
