- **Conventions, written down and held in one place.** CONTRIBUTING's naming rules gain an error suffix rule (`*Refused`: MIME
  declined an input, with a code the user can fix; `*Error`: something failed that is not the user's input) and restate
  "one name, one concept" with the `*Shape` example; `mime scaffold plugin`'s README says both. The three packers (`mime
  pack` of a saved folder, of a study folder and of a block folder, and of a publishing folder of studies) make their
  output-outside-the-folder, stray-file, name-collision and size checks through one function each; and `mime.toml`'s file name,
  format and block tables are defined once. Every refusal code and message is as it was. `mime.errors.KEPT`, the number of
  full-traceback files kept, is internal now (it was plumbing, and shared its name with the retired-version maps of
  `mime.interactions`, `mime.fields`, `mime.scene` and `mime.dynamics.body`).
