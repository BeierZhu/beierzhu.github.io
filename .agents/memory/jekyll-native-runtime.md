---
name: Jekyll native runtime
description: Why MiniRacer may fail even when the C++ runtime exists on disk.
---

MiniRacer's isolated V8 loader requires the Nix GCC runtime directory in its library search path. Keep that path scoped away from ImageMagick subprocesses: the GCC runtime available to the Ruby process can be older than the C++ ABI required by ImageMagick's libraries.

**Why:** The system library listing and native dependency inspection can resolve libstdc++, while loading MiniRacer from Ruby still fails. Installing GCC alone does not make its library visible to that loader. Exporting that runtime path for MiniRacer can make ImageMagick load an incompatible libstdc++.

**How to apply:** Keep the runtime lookup dynamic through GCC rather than hardcoding a Nix store hash. When Jekyll spawns ImageMagick, invoke it through a narrow wrapper that removes `LD_LIBRARY_PATH` for that child only. Verify MiniRacer by evaluating JavaScript and verify real image conversions through the same Jekyll launch path.
