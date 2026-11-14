# Sharing a part of an experiment: blocks

A **block** is a published, versioned part of an experiment that another lab reuses: an apparatus, a vessel network, a
vendor's probe model. It is data, never code: its things, laws, observations, values with their reasons and at most one
controller, written in the saved document's own vocabulary, with **sockets** its user binds to things of their own. The
code its entries run is in plugins its `[requires]` names, so its author does not maintain them. MIME's lab fixture holds
one, `lab_fixture/blocks/nudged_probe/`: the labx lab's probe, pushed by the lab's own nudge, with a drag and a spring,
coupled to a medium you bind.

## The folder

<!-- mime-fragment: a folder's layout, not code -->
```
nudged_probe/
  mime.toml          [block] (id, version, licence), [requires], [sockets], [[validity]], [cases], and [record]
  block.mime.yaml    the fragment (mime.fragment/1): the document's sections, a socket written {socket: <name>}
  records/           the material records the fragment names by sha256, where it names any
  descriptions/      the descriptions its entries read, likewise
  assets/            the author's copies of open or conditional asset files, where it has any
  cases/<case>/      saved experiments that use the block
```

- **`[block]`** names it, `<namespace>.<name>` and a version as Python's packaging reads one, and licenses its content
  (the fragment, records and descriptions) with an SPDX expression.
- **`[requires]`** names the distributions its fragment's entries come from, each with a range written as a
  `pyproject.toml`'s dependencies write one; `mime-engine` is always one. MIME's own development releases count as in a
  range.
- **`[sockets]`** declares what you bind: each socket's kind is `structure`, `medium`, `body` or `apparatus`. A socket on a
  frame of yours is refused in this release (`block.socket`): a block's things are placed in the world until one apparatus
  can be mounted on another's frame.
- **`[record]`** is the record's metadata (title, creators, keywords, what to cite with it). Editing it is never a new
  version; editing anything else is.

The fragment is the saved document's sections, `things`, `interactions`, `observations`, `controller`, `declared`, `ranges`,
`sources`, `materials` and `terms`, each written as a saved experiment writes it, and nothing of the experiment that uses
it (its run, scene, profile, conditions and licence). Where a binding or a placement names the user's thing, it names a
socket: `medium: {socket: medium}`, or `{socket: tube, path: lumen}` for a region of it. Names are the block's own, so an id
is written with the socket's name, `toy.couple:probe`. A socket is never a key: the keys of `declared`, `ranges` and
`sources` name the block's own things, so one that begins with a socket's name is refused (`block.socket`), since the user's
thing is the user's to declare, and a key that is not text is refused (`block.fragment`).

The block's **content hash** names its definition (`mime.toml` less `[record]`, parsed, so comments and order do not
count, its assets' locations left out) and the bytes of its fragment, records and descriptions. Its cases and its assets'
bytes are outside it.

## Using one

`exp.use(<block folder>, <socket>=...)` binds each socket to a handle `add_*` returned or a thing's name, and adds the
block's declarations to the experiment:

<!-- mime-include: examples/blocks.py#use -->

Every refusal comes at once, before anything is added:

- `block.requires`, the environment's to fix: a requirement not met here, with the line that installs what is missing.
  MIME installs nothing.
- `block.socket`: a socket unbound, unknown, bound to a thing of another kind or to one the experiment does not declare.
- `block.entry`, the block author's to fix: an entry from a distribution its `[requires]` does not name.
- `block.name`: something the block provides is already declared here. `name="<name>"` names the use and names the
  block's things `<name>_<thing>`, which is also how one block is used twice.
- `block.fragment` and `block.format`: a fragment that is not the document's shapes, or not of this release's form.

A value declared after the use on something the block provides replaces the block's, in a script as in a notebook, with a
warning. A declaration removed is treated the same. Each is recorded in the use's **overrides**. The saved document records
each use under `blocks`: the block's id, version and content hash, its tier and licence, how its sockets were bound, what it
provides and what was declared over it. The example prints:

<!-- mime-output: examples/blocks.py -->
```text
the block provides 'probe', and binds its socket medium to 'water'
exp.declare(...) declares the value probe.mass again, not the same as before: it replaces the one block labx.nudged_probe 0.1.0 provides (the use 'nudged_probe'), recorded among the use's overrides
blocks:
  nudged_probe:
    id: labx.nudged_probe
    version: 0.1.0
    content: 625c8cf8621b53d4f69456df78426f541bb46ccd70b500db0c6ab775a7a6fbd4
    tier: copied
    licence: CC-BY-4.0
    sockets:
      medium: water
    names:
      things:
      - probe
      interactions:
      - labx.nudge:probe
      - toy.couple:probe
      - toy.drag:probe
      - toy.spring:probe
      declared:
      - probe.mass
    overrides:
    - probe.mass
mime check: 0 findings, 0 to do
with the block's folder gone: input hash: reproduced exactly
```

## Tiers

The licence decides what your saved document holds, and you never write it yourself:

- **Copied** (an open or conditional licence): the block's declarations are in your document, so a load never needs the
  block, and a newer version of it changes nothing you saved. Your document's own files then hold the block's content, so
  their verdict is the AND of your licence and the block's (`mime card --notice` says so). A non-commercial block makes them
  non-commercial. MIME judges no compatibility between licences, so two different conditional licences leave the verdict
  unknown and say why. Where you declare over a value of a block whose licence carries a no-derivatives condition, the
  NOTICE and the card name the block, its condition and what you overrode; MIME judges nothing of whether that makes your
  experiment a derivative.
- **Referenced** (any other licence): the block's files stay out of every bundle you share, and your document records the
  use alone. This release refuses such a use (`load.referenced-block`); a later one reads its container. Referenced keeps the
  block's files out, but not its numbers: the values its fragment sets reach every user's saved lock and what their builds
  compute. `mime check` tells such a block's author so: what must stay private belongs in an asset with no location (a file
  had by request), never in the fragment.

The last region of the example deletes the block's folder and loads the experiment it saved, which reproduces its input
hash:

<!-- mime-include: examples/blocks.py#without -->

## Making one

Start from a saved experiment. `mime.blocks.fragment_from(<saved folder>, medium="water")` writes a fragment from its
document, with the things you name made sockets. A law or an observation that binds only those things is yours and is
left out, and so is a value over them. Write `mime.toml` beside it. Then save a case: an experiment that uses the block,
saved into `cases/<case>/`:

<!-- mime-include: examples/blocks.py#case -->

Two commands read a block folder offline, building, downloading and installing nothing:

- **`mime show <block folder>`** says what the block is and needs. That is its id, version, licence and tier, its record's
  title and creators, and its content hash. Each requirement is shown against what is installed, with the install line
  where one is not met. Then its sockets, its assets with their tiers, and what it adds to an experiment, computed from the
  fragment. `--json` gives the same as data.
- **`mime check <block folder>`** checks every key of `mime.toml`, the fragment and its sockets, `[requires]` against the
  fragment's entries and against what is installed, and each case. A case saved before the content changed is
  `block.case-stale`: save it again. A block with no case is a to do (`block.no-case`), because packing it will refuse
  it. The profile is exploratory unless `--profile` names another; research makes the to-do list findings.

A block's container (`.mimeblock`), pulling one by its DOI, verifying its cases and publishing it come in a later
release.
