# The inlined viewer

`model-viewer.min.js` is Google's `<model-viewer>` 4.3.1, the file `dist/model-viewer.min.js` of the npm package
`@google/model-viewer@4.3.1`, byte for byte: the same bytes from the npm tarball (whose sha512 is the registry's
integrity), from jsDelivr and from unpkg. `mime.viz` inlines it into a page so that the page needs no network (`mime
visuals web`), and pins it by its sha256 and by the integrity a jsDelivr page holds it to (`mime.viz._web`, held by
`tests/viz/test_gltf_web.py`).

The bundle holds, by its own licence comments and its package's dependencies, these parts, each with its licence's
text here, as its npm package ships it (`tests/viz/test_gltf_licence.py` holds each text's sha256, and that the bundle
names no licence beyond them):

| part | licence | text |
|---|---|---|
| model-viewer, Google LLC | the Apache License, as MIME's own | `LICENSE.model-viewer` |
| three.js r183, the three.js authors | MIT | `LICENSE.three` (from the npm package `three`) |
| lit (lit-html, lit-element, reactive-element), Google LLC | BSD, three clauses | `LICENSE.lit` (from `lit-element`) |
| fflate, Arjun Barrett | MIT | `LICENSE.fflate` |
| gainmap-js, MONOGRID (bundled with no licence comment) | MIT | `LICENSE.gainmap-js` (from `@monogrid/gainmap-js`) |

Each page MIME writes names them in its credit line, and `NOTICE` names them. To update: take the new release's
`dist/model-viewer.min.js` from npm, check it against jsDelivr's, read its licence comments and its dependencies
again, and move the pins, the texts and this table together.
