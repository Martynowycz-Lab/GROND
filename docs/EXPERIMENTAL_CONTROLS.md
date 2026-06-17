# Experimental Controls

GROND keeps a few lower-level controls that are useful for development, regression tests, and method audits, but they are not part of the production `GROND auto` decision surface. The most important example is the corrected-HKL `q` exponent. Internally the row-wise dynamical correction is

```text
q_i = Idyn_i / Iborn_i
Icorr_i = Iobs_i / q_i
sigma_corr_i = sigma_i / q_i
```

The production workflow uses that full correction, which is equivalent to `lambda_q = 1.0` in the diagnostic form `Icorr_i = Iobs_i / q_i^lambda_q`. Earlier experiments allowed values below one to damp the correction. That can be useful when stress-testing whether an endpoint is sensitive to overcorrection, but it is too easy to interpret as an arbitrary fit knob. For that reason `GROND auto` hides and pins the exponent to one. The lower-level `run`, `raw-polish`, `scale-replay`, `calibrate-endpoint`, and legacy engine commands may still expose `--lambda-grid` or `--lambda-q` for explicit diagnostics, and their defaults now also use the full correction.

This separation is intentional. Production reports should be explainable from the row-level Bloch and Born predictions without asking a user or reader to justify a fractional dynamical correction. Diagnostic runs can still vary the exponent when testing a hypothesis, but those outputs should be labeled as sensitivity checks rather than promoted publication endpoints.
