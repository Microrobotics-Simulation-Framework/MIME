"""A study of script A: the field's frequency swept at four values below step-out (from rest the body stays locked to
43.25 Hz), each point the experiment with its frequency declared, run together as one batched scan. The user guide
(``docs/guide/studies.md``) includes this file by its regions and prints its output, which
``tests/docs/test_tutorials.py`` and ``mime docs check`` regenerate. A short run, so the speeds it prints are a short
run's, not the reference window's."""

from script_a import script_a

from mime import Study
from mime.study import grid
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
