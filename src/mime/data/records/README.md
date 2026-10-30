# `mime.data/records`

**Copies of OpenMaterica's example records**, so that script A's record arm runs from an installed MIME. They are
examples, not reviewed library data: OpenMaterica's curated library does not exist yet.

| File | Record | Packaged as | SHA-256 of the file |
|---|---|---|---|
| `water.json` | `example/water` 1.0.1, content hash `sha256:bafb89c2...c914`: liquid water's density and dynamic viscosity at 293.15 K and 310.15 K, evaluated from the IAPWS formulations R6-95(2018) and R12-08; each Newtonian label with its `label_basis` (`inferred`) | `openmaterica_example_water` | `36add487...2761` |
| `ndfeb_n45.json` | `example/ndfeb_n45` 0.1.0, content hash `sha256:ca2a4e1a...850d`: sintered NdFeB grade N45's remanence, 1.32 to 1.37 T, a vendor's grade limits | `openmaterica_example_ndfeb_n45` | `40ff82e8...c4b0` |

## Origin

Byte for byte the file `examples/records/water.json` of [OpenMaterica](https://github.com/OpenMaterica/OpenMaterica)
at commit `eb722a9b92187df57d6010c7b7423e4f2090c3c2` on its `main` (its PR #109: each Newtonian label gains its
`label_basis`; no value, condition or label changes, so the record's version moves by a patch), and the file
`examples/records/ndfeb_n45.json` at its tag `openmaterica-v0.1.0.dev3` (commit
`1e4593c23af14d28c6f6e204c192ed4ce694d7be`), where it was last changed in commit `e37b1def`. Each record's `card` says what it holds, and each value's `provenance` names its
source.

## Licence

The records' values are licensed **CC0-1.0**, and quoted evidence is excluded from that licence, as each record's
`licence` member states (`{"values": "CC0-1.0", "evidence": "excluded"}`). They are not covered by MIME's
Apache-2.0 licence.

## The rule

These files are never edited here. A record is pinned twice: by the SHA-256 of the file's bytes, which `mime.data`
checks on every read, and by its OpenMaterica content hash, which a read checks. A changed file is refused. A newer
copy is a new file with its new pins, taken from OpenMaterica, never edited in MIME.
