# GROND Workflow Figures

This directory contains LaTeX/TikZ source for the public GROND workflow diagrams.
The master file is `grond_workflow_figures.tex`. Compile it from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error grond_workflow_figures.tex
```

The figures are intentionally generic. They describe the GROND data flow from the
standard pre-GROND structure-solution path through the recommended GROND intake,
automatic decision stages, Bloch/model-update loop, forensic gates, maps, and
endpoint outputs.

The diagrams are written for mixed audiences. A "row" means one measured spot
observation from the integration table; a "merged reflection" means many rows
have already been combined. Most GROND decisions happen before merging because
geometry, thickness, coordinate polarity, and weak-row behavior are easiest to audit while row
identity is still present.
