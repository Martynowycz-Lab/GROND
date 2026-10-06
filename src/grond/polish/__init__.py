"""Dynamical (DiffBloch-style) final polish of a finished GROND run.

The structure (fractional coordinates and displacement parameters; occupancies fixed), each crystal's
thickness and its frame scales are refined by Gauss-Newton/Levenberg-Marquardt against the unmerged work
rows with the run's own Bloch physics.  The gradient of every row intensity is exact: the Bloch-wave
propagator carries an analytic adjoint (Daleckii-Krein divided differences of the matrix exponential), and
the structure-factor chain from the parameters to the Fourier coefficients U_g is differentiated by torch
forward-mode autograd.  Free rows are predicted and reported, never fitted.
"""
