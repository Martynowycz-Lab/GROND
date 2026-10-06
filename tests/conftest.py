"""Test-suite defaults.

Physics audits compare against analytic results at ~1e-12, so the suite runs the
Bloch engine in complex128 unless a test asks for complex64 explicitly.
Production and benchmark runs keep the complex64 default.
"""

import os

os.environ.setdefault("GROND_PRECISION", "float64")
