# Data Policy

The public GROND repository is limited to installable package source, tests, documentation, synthetic examples, schemas, and release tooling. It does not distribute experiment workspaces, dataset-derived benchmark outputs, raw diffraction images, private structure files, licensed program files, article-drafting material, or complete run directories. The files under `examples/` are small synthetic input shapes with fictional paths.

Development experiments and hardware benchmarks belong in separate private or purpose-built research workspaces. A result may enter this repository only after it has been reduced to dataset-neutral package code, a focused regression test, or documentation that does not expose the originating sample or filesystem.

Before committing a result, remove local absolute paths, account names, credentials, unpublished sample identifiers, and data that cannot legally be redistributed. A compact public record should contain the command, software version or commit, input hashes, numerical configuration, a readable summary, and only the derived tables needed to support the stated result. Large or reusable datasets belong in a versioned scientific data archive with their own access terms and citation.

GROND outputs belong to the person or organization that supplied the inputs. The MIT software license does not grant permission to redistribute third-party data, structures, or external-program outputs. Each lab remains responsible for its data-use agreements and for deciding which derived products may be shared.

Generated caches, run directories, NumPy arrays, logs, raw-data folders, foreign-project checkouts, and local environments are ignored by the repository defaults. [PROVENANCE.md](PROVENANCE.md) describes the clean-source boundary for the code itself.
