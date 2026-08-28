"""Building skills.

Two stages, in order:

  mining/  read real rollouts, split them into a wrong set (the material) and a
           correct set (the false-positive control), and report which error
           families have a checkable claim surface left free.

  forge/   turn that into candidate PFs: cluster -> propose -> screen -> emit,
           then probe and measure. Only a measured accuracy change admits one.

Nothing here runs at inference. The skills it produces live in `skills/`, the
runtime that executes them is `src/skills_agent/`, and the checkers they call
are in `anchor/`. `evolving/` does the same job inside a training loop.
"""
