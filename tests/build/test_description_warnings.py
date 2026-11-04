"""An arm's description is read several times in one build, and its warnings (an element not used, one read as
massless: the build report's lines) are given once each, at the line that built the experiment, never once per read
at MIME's own line; and ``mime``'s commands show them in MIME's short form, once each however many times the command
builds (a load builds to check its lock, and a run builds again), with no source line under them.

The controls: the warnings are there to give (the AR4's file holds visuals and collisions it does not use), so a build
that gave none would fail; and the short form passes any other warning to Python's display as it was."""

from __future__ import annotations

import io
import warnings

from chain_cases import kinematic_rpm

from mime.cli._display import short_description_warnings
from mime.core.multibody import DescriptionWarning


def test_a_build_gives_each_of_its_descriptions_warnings_once_at_the_line_that_built_it():
    experiment = kinematic_rpm("described")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built = experiment.build()
    given = [w for w in caught if issubclass(w.category, DescriptionWarning)]
    texts = [str(w.message) for w in given]
    assert texts and len(texts) == len(set(texts)), "each once, where every read gave each before"
    assert all(w.filename == __file__ for w in given), {w.filename for w in given}
    assert any("<visual>" in text and "not used" in text for text in texts)
    assert all(text.startswith("rpm.arm: ") for text in texts), "named by the segment"
    reported = "\n".join(built.report.text().splitlines())
    assert all(text.removeprefix("rpm.arm: ") in reported for text in texts), "the build report's lines"


def test_the_commands_show_each_description_warning_once_in_mimes_short_form_and_others_as_they_were():
    stream, others = io.StringIO(), []

    def elsewhere(message, category, filename, lineno, file=None, line=None):
        others.append((str(message), category))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = elsewhere
        with short_description_warnings(stream):
            for _ in range(5):
                warnings.warn("rpm.arm: ar4.urdf: 18 <visual> elements not used", DescriptionWarning)
            warnings.warn("drag:umr: a range warning", UserWarning)
        warnings.warn("rpm.arm: after the command", DescriptionWarning)
    assert stream.getvalue() == "mime: warning: rpm.arm: ar4.urdf: 18 <visual> elements not used\n"
    assert others == [("drag:umr: a range warning", UserWarning), ("rpm.arm: after the command", DescriptionWarning)]
