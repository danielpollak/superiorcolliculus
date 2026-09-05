"""superiorcolliculus.behavior -- shared behavioral guidance-law analysis code
for the Mouse Hunt / Superior Colliculus prey-capture research program.

Unlike analysis.py/processing.py/viz.py/ephys.py (project-agnostic
utilities), the modules in this subpackage are schema-coupled to that
project's tautau_df DataFrame convention (columns like `avi`, `event ind`,
`Phi`, `alphadot`, `thetadot`, `tau`, `tau1`) or to its bout-dict convention
(pursuit_rollout.py) -- they encode this specific research program's data
shape and scientific hypotheses (the PN/PP/CATD/iPN guidance laws), not
generic tools. Housed here (rather than as generic-looking top-level package
functions) so that boundary stays visible. Not star-imported into the
top-level `superiorcolliculus` namespace -- import submodules explicitly,
e.g. `from superiorcolliculus.behavior import guidance_law_fitting as glf`.

Not included here: `weak_sindy_pursuit.py` stays in the main repo
(analysis/behavior/) -- it's a CLI script with a hard `pysindy`/`kneed`
dependency and a matplotlib-backend side effect on import, which would leak
onto every consumer of this package.
"""
