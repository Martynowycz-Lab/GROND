# Provenance

GROND is an independently written open-source implementation of a Bloch-wave and N-beam-style dynamical electron-diffraction workflow for integrated diffraction data.

The code in this repository was written from public scientific information and from our own experiments with open file formats and openly available example data. The implementation uses published Bloch-wave and electron-scattering theory, public documentation, public repository material, public/example datasets, and ordinary interoperability testing against files produced by crystallographic programs. No proprietary source code, decrypted code, confidential implementation notes, private partner documentation, or non-public algorithmic descriptions from any external commercial or academic crystallographic package were used to write GROND.

GROND is an independent research software project. It is not distributed by, endorsed by, certified by, or affiliated with any external crystallographic software project, company, or maintainer group. Names of external programs appear only to describe file formats, optional command-line interoperability, reporting conventions, citations, or scientific comparison points.

Several ideas implemented in GROND are standard in the electron-diffraction and crystallographic literature: Bloch-wave propagation, N-beam support selection, excitation-error calculations, thickness and orientation integration, coordinate-polarity-alt row diagnostics in acentric structures, corrected-HKL endpoint generation, least-squares endpoint checks, and merged/unmerged residual reporting. GROND reports its own row/MR residual tables with the equations written directly in the output; external program names are used only for executable configuration, file-format interoperability, citations, or conventional crystallographic comparison points.

GROND was built because there was no generally usable, installable, open-source workflow for applying row-level N-beam Bloch calculations directly to XDS- and DIALS-style integrated data. The package is intended to make the method auditable, reproducible, inspectable, and modifiable by scientific users.

## What Was Used

GROND development used:

- Publicly available crystallographic and electron-diffraction literature.
- Public documentation for relevant file formats and least-squares conventions.
- Public DIALS documentation and public DIALS source code for reflection-table and export interoperability details.
- Public or user-owned XDS/DIALS/CIF/RES/HKL inputs.
- Public/example datasets when available for understanding expected inputs, outputs, and reported parameter families.
- Open Python packages listed in `pyproject.toml`.
- Optional external executable calls only through documented command-line interfaces.

## What Was Not Used

GROND development did not use:

- Proprietary or confidential source code.
- Encrypted/decrypted private modules.
- Vendor-only implementation notes.
- Private development branches or private communications that disclosed implementation details.
- Reverse-engineered internal binaries.
- Any promise or claim that GROND reproduces the private behavior of a closed-source scattering or model-update engine.

## Clean-Room Expectations

Contributors should keep this boundary intact. Do not copy code from proprietary software, do not paste confidential documentation into issues or pull requests, and do not use reverse-engineered private internals to implement GROND features. If a future contribution is based on a paper, public repository, public documentation, or a public file-format example, cite that source in the code comment, documentation, issue, or pull request where it matters.

## License

GROND is released under the MIT License. MIT is a permissive open-source license: it allows use, modification, redistribution, sublicensing, and commercial use, provided the copyright and license notice are preserved. If a non-commercial or copyleft release is desired later, that would be a different licensing decision and should be made explicitly before accepting outside contributions.
