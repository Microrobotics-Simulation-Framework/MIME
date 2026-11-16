- **A command port's bounds and slew as study axes** (`mime.study`, `AxisKind.PORT`; G14a). `<apparatus>.ports.<command>.bounds`
  (one quantity for a symmetric pair, or a pair), `.low`, `.high` and `.slew`, and `[i]` for one element where the port
  declares one per element, are axes of a study, a run's override and a range: a point writes the port's number, its
  reason the port's own followed by the point's, and the plan judges the port at each point. The stage already held each
  number as a leaf; a controller inside the compiled step whose allocation (`ctx.allocator`) reads a declared bound or
  slew now holds it as a leaf of its node too (`port_<apparatus>__<command>__<number>`), and one axis writes both. A sweep
  of a current limit is one group and one compile, and `Study.gradient(..., wrt=["coils.ports.currents.high"])` passes
  through the stage and the allocation alike (what a stronger amplifier buys), declared a jump where the port trips. A
  number the port does not declare is refused (`study.port-undeclared`, the route to declare it). A controller called on
  the host makes its law from each member's declared bounds, as built, with no derivative through it. Not built: a
  controller believing a limit other than the hardware's. **The derived hash of every in-graph allocating loop over
  declared bounds moves** (the controller's node holds the port's numbers as leaves where it held them as constants); its
  outputs at the declared bound are those of the release before it.
