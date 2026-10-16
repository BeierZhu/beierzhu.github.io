---
name: Jekyll native runtime
description: Why MiniRacer may fail even when the C++ runtime exists on disk.
---

MiniRacer's isolated V8 loader requires the Nix GCC runtime directory in its library search path.

**Why:** The system library listing and native dependency inspection can resolve libstdc++, while loading MiniRacer from Ruby still fails. Installing GCC alone does not make its library visible to that loader.

**How to apply:** Keep the runtime lookup dynamic through GCC rather than hardcoding a Nix store hash. Verify MiniRacer by evaluating JavaScript, not merely by inspecting linked libraries.
