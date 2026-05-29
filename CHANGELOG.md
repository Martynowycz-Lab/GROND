# Changelog

## 0.2.0

This is the first GitHub-facing package layout. It reorganizes the public Python code into input/output, Bloch, refinement, map, forensic, workflow, and vendored-engine namespaces; removes dataset-specific helper scripts from the package; adds package standards files; and rewrites the user documentation for a general lab audience.

The automatic endpoint path now promotes SHELXL's recommended `WGHT` into the active pre-`END` input on every endpoint polish cycle, runs five endpoint cycles by default in full mode, keeps the default endpoint-B grid at `0.0`, and lets full auto screen bounded detector-position scale grids. Nonzero endpoint-B values remain available as explicit diagnostics, but they are no longer a production default because they can improve endpoint R factors while distorting ADPs and maps.
