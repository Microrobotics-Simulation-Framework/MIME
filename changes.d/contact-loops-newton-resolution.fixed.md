- **The contact loops' Newton property (`tests/oracles/test_contact_loops.py`) judges a group only where 0.4.0 resolves it
  against its own rounding.** 0.4.0's resolution is also taken from 64 copies of its search whose every step is moved by the
  rounding its solve may leave, which a pose's last place does not reach (a float32 FL-9 draw ended 1 mm shallower on CI). A
  group 0.4.0 does not resolve is counted, not judged: every judgement of such a group tried (its converged flag, its residual
  against the arms' or against its own rounding floor) flagged the clean search on CI's runners. Each planted defect (a short
  loop, a stale carry, an early stop) has a pinned control in both dtypes, and the property's measured per-run detection is
  stated. A test change; the contact is unchanged.
