# Inequality constraints

An inequality constraint keeps a body on the free side of a surface: a body that stays inside a tube. It is not
delivered by an edge. **The receiving dynamics takes it when it is constructed, and applies it after its own step**, so
the constraint sits inside the body's dynamics node and no graph edge carries it.

**What happens between the nodes.**
1. A law (the contact, for a surface the body must not cross) plans a confinement in its `plan`: a body role, a surface
   role, a margin, a velocity rule, and either the label of the body's extent among its values (the centre is held by
   its extent) or a spacing (the points of the body's hull are held, its orientation read).
2. The build resolves the surface to a surface of a **fixed** structure. A surface of a body, or of anything with
   dynamics, is refused (`confinement.moving-surface`): a constraint across two moving things is not built.
3. The body's dynamics must accept the target, else `target.not-accepted`. To hold the hull it must also be one that
   holds a hull (`confinement.hull-not-held`), the body must have a hull whose kind samples its surface
   (`confinement.no-hull`), and the surface's shape must evaluate many points at once and, for a kind with a support
   query, give a curvature (`confinement.surface-shape`). A spacing too coarse for the margin, or so fine that judging
   it would pass what the plan holds in memory, is refused with the spacing that would pass (`confinement.spacing-short`).
4. **At most one confinement binds a body**: sequential projections depend on their order, so a second is refused
   (`confinement.more-than-one`), naming each law that planned one. The velocity rule must be one of the built rules
   (`confinement.rule`).
5. The resolved confinement is handed to the dynamics at construction, as the shape and parameters of the surface, the
   margin, the extent or the hull's quadrature, and the rule. The dynamics projects the body back inside after its step
   and changes its velocity by the rule. How, and what the projection costs, is that dynamics' node page.
6. A post-build check refuses an edge into a target of this kind (`post-build.not-by-edge`), so a law that tried to
   deliver one by an edge fails there if it was not already refused at plan time. No test names this code, so this step
   is read from the source (`_checks.py`) and no test shows it.

**What this page does not say.** What a projection preserves and what it does not (an impulse, an energy, the
orientation of a turning body) belongs to the dynamics that applies it and to the law that planned it. The contact's own
page says what it holds; the validity of a held hull is stated where its sampling is, on the geometry kind's page.
