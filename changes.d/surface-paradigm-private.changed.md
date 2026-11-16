- The small paradigm packages' internal names leave their public surface (SURFACE-REDUCTION): 30 names that only MIME's own
  modules and tests used are out of the `__all__` of `mime.units` (`SAY_THE_ANGLE`, `parse_value_unit`,
  `rate_symbol`, `says_its_angle`, `unit_name`), `mime.things` (`DERIVED`, `Derived`, `PartMass`, `Thing`, `WORLD`, `derive`,
  `did_you_mean`, `magnetisation`, `prim_path_of`), `mime.fields` (`CALIBRATIONS`), `mime.dynamics` and its subpackages
  (`PRODUCTS_TAKEN_AS_NONE`, `RigidInertialExplicit`, `RigidInertialMoments`, `STEPPERONLINE_17HS19_2004S1`, `actuator`,
  `chosen_scheme`, `registered_actuators`), `mime.interactions` and its modules (`CENTRE_SWITCHES`, `HULL_SWITCHES`,
  `UniformTube`, `uniform_tube`), `mime.drives` (`as_drive`) and `mime.vocabulary` (`ENERGY` and `POWER`, which are
  `mime.units`'s), with no shim (the soft-tissue plugin imports none of them). A name that left is now internal: MIME's own
  modules and tests import it from the module that defines it, and no script needs it.
  Kept public because a user annotates them or MIME's own messages teach them: `mime.units.Dimension`, `Marker`,
  `Values`, `ZERO_CELSIUS` and `mime.things.Port`; `mime.units.Magnitude` and `AngleCount` (what `Quantity.value` and `Quantity.angle`
  hand back) are now in `mime.units.__all__`. `tests/test_signature_types_are_public.py` reads these six packages too, and
  counts a class a public submodule exports (`mime.dynamics.flow.RK3Scheme`) as exported.
