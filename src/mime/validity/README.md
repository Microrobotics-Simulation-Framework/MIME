# `mime.validity`

**What a result can be trusted for.**

## What goes here

- **Validity lines**: one per assumption, each with its covered range.
- **Watches**: run-time checks that report and never halt.
- **Evidence** in the build report.
- **MIME's profile lines**, for `exploratory`, `research` and `strict`, merged with OpenMaterica's.
- **The stamp** a build carries, and `exp.status()`.

## The rule

A model outside its validated range is reported, never silently switched for another. `strict` holds a higher
evidence standard. It confers nothing by itself.

## Status

Empty. It lands in a1. Until then the build's own refusals cover what dev1 needs.

Phases and releases are explained in the [package manifest](../README.md#status).
