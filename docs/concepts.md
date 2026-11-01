# Concepts: the page types, and how they link

MIME's documentation has four page types (entry, node, family and mechanism) and one idea that is a section of a page
and not a page (the rung), because the things a user needs to know do not fit one shape. A
family of laws can have several models (a surrogate, an MRS solve, a penalty law), and the model is
where the physics is, but it is not what a user writes. So there is no page per model, and no page per law: each fact
has one page that owns it, and the links between pages are generated from the registry, never typed.

## Entry

An **entry** is a registered thing a user writes in a script or a saved document: an interaction, a dynamics, a field
provider, an observation, a drive, a controller or a kind of geometry. Its page is for users. It says what the entry
models and when not to use it, what it takes (its settings, with their dimensions), what it reads and delivers, what
its derivative is and where it is not exact, and what it rests on. The hand-written part of an entry's page is short;
the rest is generated from the entry's own declarations, so it cannot disagree with the code.

## Rung

Some entries have **rungs**: one law at several levels of fidelity, chosen by the entry's `rung` setting, with no
default. A rung is a section of its entry's page, not a page of its own. A rung names the nodes it builds, and says so
in one declaration on the entry, `builds`.

## Node

A **node** is the code that computes: a MADDENING node, built by an entry. Its page is for lab authors and reviewers.
It owns the equation, in the node's `NodeMeta`, and a symbol table that joins each symbol of the equation to the code
that implements it, with its units. An equation is printed on one page only; every other page links to it. A node that
declares no `NodeMeta` has a page that says it is undocumented, and the count of such nodes is printed.

## Family

A **family** collects the laws that answer one question. Its page owns the contract every member must satisfy (a sign
rule, the settings that stand for a material's properties, the pairs a two-way law delivers equal and opposite) and the
comparison of the members that can replace one another. A cell of the comparison is a value or the words "not
measured", never blank.

## Mechanism

A **mechanism** is what happens between nodes. There is one page for each kind of target a dynamics accepts (a load,
a kinematic constraint, an operator, an equality constraint, an inequality constraint, a registration). It says how
deliveries into that kind of target combine, which interfaces are of the kind, which dynamics accept it and which
entries deliver into it, and it refers to the node equations without repeating them.

## How they link

An entry names its family, and lists the nodes each rung builds. A node lists the entries that build it. A family lists
its members. A mechanism lists the entries that deliver into its targets and the dynamics that accept them. The
vocabulary page defines every interface name the others use, and the index is a front door by goal.

Each of those links is generated from the registry export, and the number rule and the citation check run over the
generated pages as well as the hand-written ones. A number a page needs is a claim record's, named by its identifier,
and the page prints it with its basis. The checks bound names, units, links and existence. They say nothing of whether
a hand-written paragraph is true or useful.
