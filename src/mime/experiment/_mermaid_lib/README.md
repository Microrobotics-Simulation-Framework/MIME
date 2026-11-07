# The inlined Mermaid

`mermaid.min.js` is Mermaid 11.17.2's single-file build, the file `dist/mermaid.min.js` of the npm package `mermaid@11.17.2`,
byte for byte: the same bytes from the npm tarball (whose sha512 is the registry's integrity, `V6K3C8EB...BBg==`) and from
jsDelivr. `mime.experiment` inlines it into the page `Mermaid.page()` writes so that the page draws the graph with no network
(`mime show <folder> --html <file>`), and pins it by its sha256 and by the integrity (sha384) a jsDelivr page holds it to
(`mime.experiment._mermaid_page`, held by `tests/build/test_graph_and_tree.py`).

It is a classic script, not the ES-module build: `integrity` holds every byte a classic script runs, where the module build
loads further files that an `integrity` attribute does not cover.

`LICENSE.mermaid` holds, for Mermaid and for each package its bundle can hold, the name, the version, the licence its
`package.json` states and the text of its licence file as its npm package ships it. The bundle is minified with no licence
comments but DOMPurify's, so the list is made from the dependency tree: every package of the production tree of
`mermaid@11.17.2` (an `npm install` with scripts off), without the packages that hold only type declarations (`@types/*`),
and the packages that Mermaid's parser (`@mermaid-js/parser` 1.2.1) bundles into its own build, as its source maps list them
(Langium, Chevrotain, lodash-es, the vscode-languageserver parts and path-browserify). That is a superset of what the bundle
holds: a package that only Node uses is listed though the bundle has none of it. The licences met are MIT, ISC,
BSD-3-Clause, Apache-2.0, Unlicense and DOMPurify's `(MPL-2.0 OR Apache-2.0)`, used here as Apache-2.0.
`tests/build/test_mermaid_licence.py` holds the file's sha256, that it names every package it should and that the bundle's
own licence comments name no licence it lacks.

To update: take the new release's `dist/mermaid.min.js` from npm, check it against jsDelivr's, read the dependency tree and
the parser's source maps again, regenerate `LICENSE.mermaid`, and move the pins, the texts and this page together.
