"""Tutorial A: de Jongh's FL-9 screw in a 1/4" tube of water, built twice, as script A's reproduction arm and then as
its record arm, each run for a short time. The user guide (``docs/guide/tutorial_a.md``) includes this file by its
regions and prints its output, which ``tests/docs/test_tutorials.py`` and ``mime docs check`` regenerate.

The experiment itself is ``examples/script_a.py``, which the guide includes by its regions beside this file.
"""

from script_a import record_arm, script_a

from mime.units import mm_per_s, s

# region: short
#: A short run. Script A's reference window is the second half of a run of five seconds; this run is one second, so
#: its speed is not that window's.
SHORT = s(1.0)
HALF = s(0.5)
# endregion short


# region: values
def show_water(label, built):
    """The values the build read for water, each with where it came from."""
    print(label)
    for value in built.report.values:
        if value.path.startswith("water."):
            print(f"  {value.path} = {value.value} {value.unit}, {value.source.split(' (sha256')[0].split(',')[0]}")
# endregion values


# region: run
def swim(label, built):
    """Run the short run, recording the swim speed's series, and read its second half."""
    result = built.run(SHORT, record=["swim_speed:umr.speed"])
    speed = result["swim_speed:umr"].speed.between(HALF).mean()
    print(f"  swim speed over the second half of a {SHORT.value:g} s run: {speed.to(mm_per_s):.2f} mm/s")
    (watch,) = result.watches
    print(f"  post-run watch on the drag: {watch.status}")
# endregion run


# region: arms
reproduction = script_a().build()
show_water("reproduction arm: every value declared, with its reason", reproduction)
swim("reproduction arm", reproduction)

records = record_arm().build()
show_water("record arm: water's values from OpenMaterica's record", records)
swim("record arm", records)
print(records.report.pending())
# endregion arms
