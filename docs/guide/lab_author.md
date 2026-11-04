# Extending MIME: a lab's own plugin

A lab adds a model to MIME the way MIME adds its own, from its own package and without editing MIME. This page says how, and
what the lab may claim for what it adds.

## What a plugin is, and what `lab_fixture` is

A **plugin** is an installed Python package that adds to MIME through entry points in its `pyproject.toml`: entries (models,
observations, dynamics, geometry kinds and the other kinds below), asset records, or adapters for where files are fetched
from. MIME finds what an installed plugin registers, and never installs anything itself. A lab's package of its own models
is a plugin; so is a package that only ships asset records.

The worked example is `lab_fixture/`, and it is not something to install or build on. It is MIME's own smallest plugin, kept
in MIME's repository so that CI builds and installs it in every lane and shows that a package outside MIME can register what
MIME registers for itself. If the fixture ever needed an edit to MIME to pass, the extension contract would be wrong;
`tests/test_lab_fixture.py` is the control that registering from outside works. Every piece of code on this page is included
from its files, so the page shows code a test runs. A real plugin has the same shape in a repository of its own, with its own
tests and versions. Publishing one (versions and compatibility, releasing it, and a template to start from) is documented in
a later release.

## What a lab registers

An **entry** is a registered, versioned thing the build resolves by name: an interaction (a law between things), an
observation, a dynamics (a chain segment is one), a field provider, a controller, a drive, a geometry kind, a wall kind. Each
kind has its `register_*` function in `mime.registry`. The fixture registers an interaction, a chain segment, an observation
and a geometry kind. It registers no field provider, controller, drive or wall kind, and no family, so there is no worked
example of those here; the contract for each is the base class of its kind.

## A package with a namespace

A lab owns a **namespace**, registered once, and names its entries in it (`labx.nudge`). A name outside a registered namespace
is MIME's and is refused from a module outside MIME. A namespace named like the first word of a MIME refusal code or of a MIME
entry is refused too, because it would take those for its own.

How a saved experiment finds the lab's entries is an entry point, in the package's `pyproject.toml`:

<!-- mime-include: lab_fixture/pyproject.toml -->

Loading imports that module and nothing the document names. The build only resolves names from installed packages, and
never installs anything. The saved document stores each entry's name and version, and the lock names the package as the
entry's distribution.

## An interaction

The fixture's interaction is a constant load on a toy body. Its node holds the state:

<!-- mime-include: lab_fixture/src/mime_lab_fixture/__init__.py::NudgeNode -->

and the entry declares what the build needs of it:

<!-- mime-include: lab_fixture/src/mime_lab_fixture/__init__.py::Nudge -->

- `name` and `version` key the entry. The saved document stores both, so a newer version is a different entry.
- A setting is a dataclass field, and a dimensioned one declares its dimension in its annotation
  (`Annotated[Quantity, Force]`). Registration refuses a `Quantity` with no marker, a marker on a setting that is not a
  `Quantity`, and a default of another dimension. A rate per second says what it counts: `Frequency` for one in Hz,
  `AngularVelocity` for one in rad/s; one that says neither is refused, as a setting, a command or an initial state,
  and so is a default of the other kind.
- `setting_roles` says what each setting is for (a design choice, a physical value, a numerical choice) and the kind of its
  default. A setting that declares nothing is required, and the build reports a written default that declares nothing as
  `undeclared`, which the exploratory profile stamps and the research profile refuses ([profiles](../profiles.md)).
- `roles` are the things the entry binds, `delivers` what it delivers into them, and `build` adds its nodes through a
  restricted builder. `builds` names the node classes each rung builds, which a check compares with what a case builds.
- `differentiability` is the entry's honest status, discussed under the claims below.

An observation registers the same way, and declares its outputs as MIME's own do, so a run's result reads them by name with
their units:

<!-- mime-include: lab_fixture/src/mime_lab_fixture/__init__.py::Height -->

A chain segment (`labx.gantry`) and a geometry kind (`labx.bead`) are in the same file. The fixture's nodes declare no
`NodeMeta` and no symbol table, which MIME's own nodes do; a node without them gets a page that says it is undocumented.

## What a lab may claim

Everything MIME prints about a lab's entry is what the entry declares, labelled as such. The lab may claim, and MIME will
print as the lab's, exactly the following; nothing else comes from MIME's tests.

- **A differentiability status.** An entry is `exact` and declares no gradient record. Otherwise it declares a
  `GradientRecord`: a `piecewise_kink` or `piecewise_jump` entry names the switches its law evaluates (its own, or its
  shape's); a `fixed_key`, `unverified` or `cut` entry states where, why and a route to exact. A value an entry leaves out
  of its record is `NOT_ESTABLISHED`: the pages print it as such and `mime docs check` counts it for a lab, and does not
  fail it, except for an `unverified` or `cut` entry, whose reason and route must be stated.
- **Validity lines.** The build reports them as *declared, not yet checked*. They say where the model holds; they are not
  evidence that it does.
- **Conformance to a family**, by running the family's suite in the lab's own CI:
  `mime.testing.assert_conforms(law, family=..., oracle=...)`. Passing it in your own CI is how you claim it (the testing README, `src/mime/testing/README.md`).
- **An oracle you name.** A check against a closed form is evidence of the model against that form. It is not evidence of
  the model against the world, and a claim should say which it is.
- **References**, as bibliography keys of the lab's own bibliography. MIME counts a lab's keys and does not check them
  against its own.

A lab may not take a MIME name or register under MIME's namespace. A bound and an estimate are different claims ([scope and limits](scope.md)): a maximum over the cases you ran is an
estimate of a bound, whatever its size.

## Shipping assets from a lab

A lab's files are asset records, the same as MIME's ([assets in depth](assets.md)):

- **Records ship in the package.** A `mime.assets` entry point names a package whose data holds JSON files of records or
  manifests, which MIME finds beside its own. An experiment that pins one keeps it when the package moves on, and two
  packages that disagree on a name are refused until an experiment pins one.
- **Small bytes go inside the package**, named `package:<package>/<path>`; large ones are hosted (an https server, a DOI)
  and named by their address. Either way the hash is the only trust.
- **A lab's own mirror** (a mounted share, an internal server) goes in each machine's locations file, never in a record, so
  it never travels with a shared experiment.
- **Storage that needs a login** is an adapter, registered with a `mime.asset_adapters` entry point and checked in the
  lab's own tests by `mime.testing.verify_adapter`. Its token is kept with `mime credentials set host:storage.example.org`
  and sent to that host only.
- **The package's CI runs `mime assets audit dist/*`.** It fails on any file whose terms let no copy be hosted, inside
  archives, compressed streams and base64 too, and on a mesh no record names. A mesh converted to another format has other
  bytes, which only the second check sees, and the lab's own records are believed for a mesh no other record knows.

## Its pages

`mime docs build` generates a page for every registered entry, a lab's included, with no work: its settings, their roles and
defaults, and what is not established. A hand-written source beside it (`entries/<kind>/<name>.md`) is printed in the page when
it exists, and counted when it does not. The index lists a lab's entries beside MIME's.

`mime registry --json` prints every registered entry and what it declares, in a fresh process, which is how to see that your
package registered what you meant.
