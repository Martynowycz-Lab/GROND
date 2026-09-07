# Provenance

GROND applies published Bloch-wave and electron-scattering theory to integrated electron-diffraction observations. Its particular workflow connects signed XDS/DIALS rows, physical geometry, nuisance regularization and conventional structural-refinement endpoints. Bloch propagation, N-beam support, excitation errors and dynamical hand sensitivity are established scientific ideas; GROND does not claim their invention.

The scientific and software sources relevant to the implementation are listed in [docs/REFERENCES.md](docs/REFERENCES.md). File-format adapters use public format documentation and interfaces. Python dependencies are declared in `pyproject.toml`; optional crystallographic executables are configured separately and are not bundled. External program names describe interoperability and comparison, not endorsement.

Contributions must identify the papers, public code and format documentation on which they depend. Code reused from another project requires a compatible license and retained attribution. Do not contribute proprietary code, confidential implementation material, or code whose provenance cannot be established. [CONTRIBUTING.md](CONTRIBUTING.md) gives the contribution rules, and [DATA_POLICY.md](DATA_POLICY.md) separates redistributable package material from scientific inputs and development workspaces.

GROND is distributed under the MIT License. That license covers this software; it does not grant rights to third-party programs, structures, diffraction data or publications.
