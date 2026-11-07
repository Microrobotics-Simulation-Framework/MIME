"""A study of script A: the field's frequency swept at four values below step-out (from rest the body stays locked to
43.25 Hz), each point the experiment with its frequency declared, run together as one batched scan; then the step-out
curve from rest, the frequency swept through step-out with the body's phase lag observed. The user guide
(``docs/guide/studies.md``) includes this file by its regions and prints its output, which
``tests/docs/test_tutorials.py`` and ``mime docs check`` regenerate. A short run, so the speeds it prints are a short
run's, not the reference window's."""

from script_a import script_a

from mime import Study
from mime.observations import PhaseLag
from mime.scene import X
from mime.study import grid, step_out
from mime.units import Hz, mm_per_s, s

# region: sweep
experiment = script_a()
field = experiment.handle("field")
study = Study(experiment).sweep(grid({field.commands.frequency: [Hz(10.0), Hz(20.0), Hz(30.0), Hz(40.0)]}),
                                duration=s(0.5), record=["swim_speed:umr.speed"])
# The plan's memory estimate is left out of this print: its bytes depend on the machine, the jaxlib and the Python it
# is measured on. print(study.plan()) shows it.
print(study.plan().text(memory=False))
# endregion sweep

# region: read
result = study.run()
frequencies = result.coordinates["field.commands.frequency"].to(Hz)
for index, frequency in enumerate(frequencies):
    member = result.member(index)
    speed = member["swim_speed:umr"].speed.between(s(0.25)).mean().to(mm_per_s)
    print(f"{frequency:g} Hz: swim speed {speed:.2f} mm/s over the second half, {speed / frequency:.3f} mm per turn")
# endregion read

# region: override
alone = experiment.build().run(s(0.5), override={field.commands.frequency: Hz(20.0)})
print(alone.overridden_text())
# endregion override

# region: step_out
observed = script_a()
observed.observe(PhaseLag("umr", field="field.magnetic", about=X, after=s(2.5)))
frequency = observed.handle("field").commands.frequency
curve = Study(observed).sweep(grid({frequency: [Hz(41.0 + 0.5 * i) for i in range(11)]}), duration=s(5.0)).run()
(bracket,) = step_out(curve, "phase_lag:umr", frequency)
print(bracket.text())
print(", ".join(f"{f:g} Hz {r:.3f}" for f, r in zip(bracket.frequencies, bracket.ratios, strict=True)))
# endregion step_out
