# The saved form

What a load of a saved experiment checks, and what it refuses. A saved folder is what `built.save(folder)` writes:
the document (`experiment.mime.yaml`, what the experiment declares), the lock (`mime.lock`, what its build resolved
and wrote), the reference outputs, the stage and the layers it references, and the record and description files the
experiment reads. `Experiment.load(folder)` reads it back and rebuilds the experiment, and `mime verify <folder>`
also reruns it against the reference outputs.

This specification is normative, and the loader is its authority: where a sentence here and the code disagree, the
code is what MIME does, and the disagreement is a defect of one of them to report. The document's and the lock's
shapes are the closed TypedDicts of `mime.saved`; the JSON Schemas generated from them, published beside this page,
are a convenience for an editor, and differ from the loader where the page's table of known gaps says.

**How to read a statement.** Each statement is a list item opening with its id in brackets and ending with its
cases: the tests that hold it, resolved through pytest's collection by `mime docs check`, or "not yet covered", which
`docs/ratchets/not_yet_covered.txt` lists until a test covers it. A statement is covered only by a test CI runs: one
whose every case is marked `skip`, `xfail`, `slow` or `gpu` is shown as citing tests CI does not run, and is listed
there too. A refused load raises `LoadRefused` with every
reason it found, each by its code, with what to do about it; nothing of a refused folder is loaded. The codes are the
refusal table's (`mime/build/refusal_table.txt`).

## The folder

- **[folder.layout]** A saved folder holds the document, the lock, the reference outputs, the stage and its layers,
  and the files of `records/` and `descriptions/`, each one level down. A lock that names a file anywhere else is
  refused before the file is read, and a file the lock does not name is refused where a bundle would carry it.
  *Cases:* `tests/sharing/test_hostile_bundles.py::test_a_lock_that_names_a_file_outside_the_layout_is_refused_before_it_is_read`,
  `tests/sharing/test_hostile_bundles.py::test_a_file_the_lock_does_not_name_is_refused_even_where_a_folder_holds_files`.
- **[folder.missing]** A folder that does not exist is refused as `load.missing-folder`, naming the saved folders
  beside the path given.
  *Cases:* `tests/errors/test_errors.py::test_loading_a_folder_misspelt_names_the_saved_ones_beside_it`.
- **[folder.no-code]** Reading a folder runs no code: the document is read with a YAML loader that constructs no
  object a tag names, the lock as JSON, and the reference outputs with pickling off. A Python tag in the document is
  refused as `load.code-in-document`, and reference outputs holding a pickled object as `load.reference`, the pickle
  never run. The stage and its layers are hashed; a stage whose bytes are not the ones pinned is read by USD, from its
  text, to compare it with the stage the rebuild writes.
  *Cases:* `tests/saved/test_document.py::test_a_python_tag_is_refused_and_constructs_nothing`,
  `tests/saved/test_load_refusals.py::test_reference_outputs_holding_a_pickled_object_are_refused_and_run_nothing`.

## The document

- **[document.unreadable]** A document that cannot be read is refused as `load.missing-file`, naming the file.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_that_cannot_be_read_is_refused_naming_it`.
- **[document.encoding]** A document whose bytes are not text in the encoding `save` writes is refused as
  `load.document-encoding`, naming the byte that cannot be read.
  *Cases:* `tests/errors/test_errors.py::test_files_that_cannot_be_read_are_refused_with_their_codes`.
- **[document.yaml]** A document that is not YAML, or that holds a mapping with one key twice, is refused as
  `load.document-yaml`. A YAML loader that keeps the last of a repeated key without a word is not this loader.
  *Cases:* `tests/saved/test_document.py::test_a_key_given_twice_is_refused`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-document-that-is-not-yaml]`.
- **[document.shape]** The document is checked against the closed TypedDict `ExperimentDocument`: an unknown key, a
  missing key and a value of another type are each refused as `load.document-shape`, naming its path. An integer is
  an integer, never a number with a zero fractional part; a boolean is neither an integer nor a number; a mapping's
  keys are strings.
  *Cases:* `tests/saved/test_document.py::test_an_unknown_key_and_a_value_of_another_type_are_refused_by_path`,
  `tests/saved/test_schema.py::test_what_the_annotations_allow_the_check_accepts_and_a_change_it_refuses`,
  `tests/saved/test_json_schema.py::test_the_readers_disagree_exactly_on_the_recorded_gaps`.
- **[document.newer]** A document of a form newer than the newest this MIME reads is refused as `load.newer-form`,
  naming the file, the form, and the MIME to install.
  *Cases:* `tests/saved/test_newer_form_saved.py::test_a_form_one_above_the_newest_is_refused_naming_the_file_the_version_and_the_advice`.
- **[document.previous]** A document of an earlier form is read under the current one, and loads under the profile
  its form had; a key that a later form added is refused in it.
  *Cases:* `tests/saved/test_previous_form.py::test_a_previous_folder_loads_as_exploratory`,
  `tests/saved/test_previous_form.py::test_the_control_a_previous_document_holding_a_key_of_version_two_is_refused`,
  `tests/saved/test_terms_saved.py::test_an_earlier_version_given_a_terms_key_is_refused`,
  `tests/saved/test_duration_saved.py::test_the_control_an_earlier_form_holding_the_key_of_version_five_is_refused`,
  `tests/saved/test_form7.py::test_an_earlier_document_holding_based_on_is_refused`,
  `tests/saved/test_form7.py::test_control_an_upgrade_that_accepts_based_on_in_an_earlier_document_is_caught_every_run`,
  `tests/saved/test_form7.py::test_a_version_six_document_that_declares_a_run_length_reads`.
- **[document.duration]** The run's length, where the experiment declares one (`Experiment(..., duration=...)`), is
  saved as `run.duration` and is part of the hashed input: a duration edited after its lock was written is refused as
  `load.input-hash`. `mime run` on the folder runs that long when it is given no `--duration`, and the option still
  overrides it; a folder with none declared needs the option, and is refused as `run.duration` without it. A duration
  that is not a finite time above zero, or not a whole number of base steps, is refused when the experiment is built,
  never saved. A document of an earlier form declares none and loads as it did.
  *Cases:* `tests/saved/test_duration_saved.py::test_a_saved_folder_runs_as_long_as_it_declared_with_no_duration_given`,
  `tests/saved/test_duration_saved.py::test_an_override_beats_the_saved_length`,
  `tests/saved/test_duration_saved.py::test_the_control_a_saved_duration_edited_after_its_lock_is_refused`,
  `tests/saved/test_duration_saved.py::test_the_control_a_folder_with_no_duration_still_needs_one`,
  `tests/saved/test_duration_saved.py::test_an_earlier_form_loads_without_a_duration_and_still_needs_one`,
  `tests/saved/test_duration_saved.py::test_the_control_a_duration_a_run_would_refuse_is_refused_at_the_build_and_never_saved`.
- **[document.based-on]** An experiment made from a saved one (`Experiment.based_on`) records it as `based_on`: its
  name, the input hash its lock records, the MIME version that saved it, and the DOI of the record its lock names,
  where it names one; the immediate one alone. It is part of the hashed input: a record edited after its lock was
  written is refused as `load.input-hash`, as any edit is. The new experiment's lock is its own and names no record;
  the folder it was made from is never written. A reader of version 6 refuses a folder of version 7 as
  `load.newer-form`.
  *Cases:* `tests/saved/test_based_on.py::test_saving_writes_a_new_folder_whose_document_records_the_parent_and_whose_lock_is_its_own`,
  `tests/saved/test_based_on.py::test_the_record_names_the_parents_doi_where_its_lock_names_the_record_it_was_published_as`,
  `tests/saved/test_based_on.py::test_the_record_edited_by_hand_after_its_lock_was_written_is_refused`,
  `tests/saved/test_based_on.py::test_an_experiment_based_on_one_that_was_itself_based_on_another_records_its_immediate_parent`,
  `tests/saved/test_based_on.py::test_the_parents_folder_is_never_written`,
  `tests/saved/test_form7.py::test_dev1s_reader_refuses_a_folder_of_version_seven_as_newer_each_file_once`.
- **[document.terms]** The document's licence, custom licences, terms and permissions are read in the form `save`
  writes them; terms in any other form are refused by their own codes.
  *Cases:* `tests/saved/test_terms_saved.py::test_terms_edited_by_hand_are_refused_at_load_by_their_code`.
- **[document.licence-unlisted]** A licence the installed packaging does not list is read as written, and the load
  says so; it is never refused for that.
  *Cases:* `tests/saved/test_terms_saved.py::test_a_licence_the_installed_packaging_does_not_list_is_read_as_written_and_the_load_says_so`.

## The lock

- **[lock.unreadable]** A lock that cannot be read, or that is not JSON, is refused as `load.lock`.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_lock_that_is_not_json_is_refused`.
- **[lock.shape]** The lock is checked against the closed TypedDict `Lock`, every key of it read only, and is
  refused as `load.lock-shape`, naming each path.
  *Cases:* `tests/assets/test_folder_assets.py::test_a_malformed_asset_record_in_a_lock_is_refused_by_the_saved_forms_reader`,
  `tests/saved/test_schema.py::test_the_lock_is_read_only_throughout`.
- **[lock.same-form]** The document and the lock are of one form of the saved form; a document beside a lock of
  another form is refused as `load.lock-shape`.
  *Cases:* `tests/saved/test_form_v2.py::test_the_control_mixed_versions_are_refused`,
  `tests/saved/test_previous_form.py::test_the_control_a_previous_document_beside_a_current_lock_is_refused`.
- **[lock.names]** Before anything of the lock is read or printed, each file it names is held to the folder's layout,
  each distribution to the names Python's packaging allows and each version to its versions; a lock that breaks one
  is refused as `load.lock-shape`, its text printed escaped.
  *Cases:* `tests/sharing/test_hostile_bundles.py::test_a_lock_naming_a_distribution_or_version_that_is_not_one_is_refused_and_prints_nothing_raw`,
  `tests/sharing/test_hostile_bundles.py::test_a_lock_that_names_a_file_outside_the_layout_is_refused_before_it_is_read`.
- **[lock.repeated-key]** A lock that holds a mapping with one key twice is refused as `load.lock`: a reader keeping
  the first value would read another lock than one keeping the last. Every module that reads a lock reads it through
  one reader, which refuses the repeat; an asset fetch refuses it as `asset.lock`, and a run's environment as
  `env.lock`.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_lock_holding_a_key_twice_is_refused_by_the_load_and_by_every_reader_of_the_lock`,
  `tests/saved/test_load_refusals.py::test_every_module_reads_the_lock_through_the_one_reader`,
  `tests/saved/test_json_schema.py::test_the_readers_disagree_exactly_on_the_recorded_gaps`.
- **[lock.newer]** A lock of a form newer than the newest this MIME reads is refused as `load.newer-form`; a document
  and a lock that are both newer are each refused once.
  *Cases:* `tests/saved/test_newer_form_saved.py::test_a_document_and_a_lock_that_are_both_newer_are_each_refused_once`.
- **[lock.publication]** A lock may name the published record that holds it (`mime push` writes it into the lock it
  packs), outside the input hash, so that `mime pack` makes the published bundle again; a lock changed between staging
  and publishing is left as it is.
  *Cases:* `tests/sharing/test_push_publish.py::test_a_production_publish_rewrites_the_lock_so_pack_makes_the_published_bundle`,
  `tests/sharing/test_push_publish.py::test_a_lock_changed_between_staging_and_publish_is_left_alone`.
- **[lock.referenced]** A file the lock says travels as a reference, and that is not in the folder, is refused as
  `asset.missing`, with the command that fetches and places it; one that is there is held to its pin as every file is.
  *Cases:* `tests/sharing/test_referenced.py::test_a_folder_missing_a_referenced_file_is_refused_with_the_command_that_fetches_and_places_it`.

- **[lock.records]** Each asset record of the lock (`assets`) is whole: beside the five keys every record has, each
  key the record states (its credit, role, kind and derivation, notice, custom licences, version, citation, and the
  package of each adapter its locations need), written only where stated, outside the input hash; its tier is never
  written, but read from its licence. A file the experiment declares by its record is held so, and the lock names only
  the record's own locations, never those a machine's locations file adds.
  *Cases:* `tests/assets/test_records.py::test_a_full_record_round_trips_and_keeps_its_kind_role_and_identity`,
  `tests/assets/test_records.py::test_control_a_record_that_writes_its_own_tier_is_refused`,
  `tests/sharing/test_records_travel.py::test_the_lock_holds_the_declared_record_whole_and_never_this_machines_locations`.
- **[lock.visuals]** The lock's `visuals` holds the manifest of each set of meshes the stage draws, by its name: their
  records, never their bytes, outside the input hash.
  *Cases:* `tests/viz/test_visual_pins.py::test_the_lock_holds_the_manifest_and_the_folder_holds_none`.
- **[lock.previous-records]** A lock of a form before the one that added them that holds `visuals` or a key of an asset record
  beyond the five is refused as `load.lock-shape`, never read as stating it.
  *Cases:* `tests/saved/test_form6.py::test_an_earlier_lock_holding_a_key_version_six_added_is_refused`,
  `tests/saved/test_form6.py::test_control_an_upgrade_that_accepts_the_keys_in_an_earlier_lock_is_caught_every_run`.
- **[lock.pins]** The lock's records are the experiment's pins: a load reads each asset under the hash its lock names,
  whatever a shared source holds under that name now, and the build report names each source that holds another
  version; one not stored is refused as `asset.missing`. A saved stage's meshes keep the manifest and sha256 it names
  likewise.
  *Cases:* `tests/saved/test_lock_assets.py::test_the_pin_wins_a_lock_naming_the_table_under_another_hash_reads_that_hash`,
  `tests/saved/test_lock_assets.py::test_a_shared_source_holding_another_version_of_the_table_is_reported_and_the_pin_kept`,
  `tests/viz/test_visual_pins.py::test_the_saved_arm_loads_with_its_stage_rebuilt_from_its_pins`,
  `tests/viz/test_visual_pins.py::test_control_a_rebuild_that_ignores_the_saved_stages_pins_is_refused_every_run`.

## The files the lock pins

- **[files.reference]** Reference outputs that cannot be read are refused as `load.reference`.
  *Cases:* `tests/errors/test_errors.py::test_files_that_cannot_be_read_are_refused_with_their_codes`.
- **[files.reference-changed]** Reference outputs whose arrays are not the ones the lock records are refused as
  `load.reference-changed`.
  *Cases:* `tests/saved/test_load_refusals.py::test_reference_outputs_whose_arrays_changed_are_refused`.
- **[files.stage]** A stage edited after its lock was written is refused as `load.stage-changed`; where only the
  derived prims were edited, the load reports it and goes on.
  *Cases:* `tests/scene/test_saved_scene.py::test_a_stage_edited_after_its_lock_is_refused_at_load`,
  `tests/scene/test_derived_prims.py::test_an_edit_beyond_the_derived_prims_is_refused`.
- **[files.layer]** A referenced layer whose bytes are not the ones pinned is refused as `load.asset-changed`.
  *Cases:* `tests/scene/test_saved_scene.py::test_a_changed_asset_file_is_refused_at_load`.
- **[files.record]** A record file edited, by one byte, or missing from the folder, is refused as
  `load.record-changed`.
  *Cases:* `tests/materials/test_pins.py::test_a_saved_record_file_edited_by_one_byte_is_refused_at_load`,
  `tests/materials/test_pins.py::test_a_record_file_missing_from_the_folder_is_refused_at_load`.
- **[files.remote-asset]** A remote asset the build read must be named in the lock; a lock without it is refused as
  `load.assets`. A lock naming it under another hash is the experiment's pin, which the rebuild reads
  (`lock.pins`).
  *Cases:* `tests/saved/test_lock_assets.py::test_control_a_lock_without_the_tables_name_is_refused_when_loaded_naming_it`,
  `tests/saved/test_lock_assets.py::test_the_pin_wins_a_lock_naming_the_table_under_another_hash_reads_that_hash`.

## Rebuilding the experiment

- **[rebuild.entry]** Each entry the document names must be registered by an installed package at the version it
  names: one no package registers is refused as `load.unknown-entry`, and one registered at other versions only as
  `load.version`, each with the line to install.
  *Cases:* `tests/saved/test_load.py::test_an_entry_no_installed_package_registers_is_refused_with_the_line_to_install`,
  `tests/saved/test_load.py::test_another_version_is_refused_with_the_line_to_install`.
- **[rebuild.namespace]** A namespace whose providing package fails to import is refused as `load.namespace-import`,
  with the line that installs the version the lock names.
  *Cases:* not yet covered.
- **[rebuild.units]** A quantity written in a unit MIME does not write is refused as `load.units`: an entry's setting,
  a thing's initial state, the controller's rate, a declared value, a condition, and the run's step.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-setting-in-another-unit]`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[an-initial-state-in-another-unit]`,
  `tests/saved/test_load_refusals.py::test_the_controllers_rate_in_another_unit_is_refused`,
  `tests/saved/test_body_side_document.py::test_a_load_refuses_an_edited_document_naming_what_to_do[a-declared-value-in-another-unit]`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-condition-in-another-unit]`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[the-step-in-another-unit]`.
- **[rebuild.settings]** A setting the entry does not have is refused as `load.setting-unknown`, naming the settings
  it has; a group of settings no installed package registers, and settings the entry cannot be built from (one it
  needs left out), as `load.setting`.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-setting-the-entry-does-not-have]`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-group-no-package-registers]`,
  `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-setting-the-entry-needs-left-out]`.
- **[rebuild.thing]** A thing written otherwise than `save` writes it (dynamics its kind does not take, a field of
  another kind, a prim moved) is refused as `load.document-thing`.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[dynamics-a-medium-does-not-take]`,
  `tests/saved/test_body_side_document.py::test_a_load_refuses_an_edited_document_naming_what_to_do[commands-on-a-medium]`,
  `tests/scene/test_saved_scene.py::test_a_document_whose_python_first_thing_moved_prim_is_refused`.
- **[rebuild.material]** A thing naming a material the document's materials do not hold is refused as
  `load.document-material`.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[a-material-the-document-does-not-hold]`.
- **[rebuild.id]** An interaction's or an observation's id is the one its registered name and bindings give; another
  is refused as `load.id`, since an id is part of the model and a renamed one is a different experiment.
  *Cases:* `tests/saved/test_load_refusals.py::test_a_document_edited_by_hand_is_refused_by_its_code[an-id-its-bindings-do-not-give]`.
- **[rebuild.input]** The rebuilt input, the document with what it resolves to, must equal the lock's exactly. A
  document edited after its lock was written is refused as `load.input-hash`, naming what changed; so is a document
  whose entries now resolve otherwise under the same names and versions, as the environment's fault.
  *Cases:* `tests/saved/test_load.py::test_a_document_edited_after_its_lock_is_refused_naming_what_changed`,
  `tests/saved/test_load.py::test_a_document_saved_before_a_version_changed_is_refused_naming_the_entry`.
- **[rebuild.version]** Development releases (0.3.0.devN) make no compatibility promise. MIME's version is part of
  the rebuilt input (the document's `mime` line and the resolved record's `vocabulary.mime`), so a folder saved by one
  development release is refused by the next as `load.input-hash`, whatever else is the same; the message names the
  versions as a change of `resolved.vocabulary.mime` and says the document was changed, which is not what happened.
  Keep the script that built the experiment and save it again with the new release. It holds of a saved folder in
  every use of it (`Experiment.load`, `mime verify`, `mime run`, `mime pack`); a bundle unpacks (`mime pull` names the
  MIME that packed it) but its folder is refused in the same way; and a run folder's outputs still read
  (`RunResult.read`) while the `experiment/` folder inside it is a saved folder, so it is refused.
  *Cases:* `tests/saved/test_load.py::test_a_folder_saved_by_another_mime_version_is_refused_and_one_saved_by_this_one_loads`.
- **[rebuild.stage]** A stage written by a stage writer this MIME does not have is refused as `load.stage-writer`; in
  the environment the lock was written in, a rebuilt stage of other bytes is refused as `load.stage-hash`.
  *Cases:* `tests/saved/test_stage_writer.py::test_the_control_a_lock_naming_another_writer_or_a_stage_neither_writes_is_refused`.
- **[rebuild.derived]** In the environment the lock was written in, the rebuilt graph must be the saved one, hash
  for hash, or the load is refused as `load.derived-hash`. In another environment its physics is compared within the
  lock's tolerance, and each number beyond it, and each static array that differs, is reported (the observations are
  not compared); there a difference in the derived graph or in the rebuilt stage is reported, never refused.
  *Cases:* `tests/saved/test_load.py::test_a_factory_that_builds_another_graph_under_the_same_version_is_refused`,
  `tests/saved/test_load.py::test_in_another_environment_differences_are_reported_and_outputs_compared_within_tolerance`.
- **[rebuild.same]** A folder saved and loaded in one environment rebuilds the same experiment, and `mime verify`
  reruns it to the reference outputs bit for bit.
  *Cases:* `tests/saved/test_load.py::test_a_saved_experiment_loads_and_verifies_bit_for_bit`.

## The schemas

- **[schema.second-reader]** The JSON Schemas are generated from the same closed TypedDicts the loader checks
  against. Over the committed folders, the planted inputs and the files drawn from the TypedDicts, a schema and the
  loader disagree exactly on the gaps the page's table records; where they disagree, the loader decides.
  *Cases:* `tests/saved/test_json_schema.py::test_the_readers_disagree_exactly_on_the_recorded_gaps`,
  `tests/saved/test_json_schema.py::test_on_drawn_files_and_their_changes_the_readers_agree`.
- **[schema.golden]** Every committed document and lock validates against its schema, as the loader reads it.
  *Cases:* `tests/saved/test_json_schema.py::test_every_golden_document_and_lock_validates_as_the_loader_reads_it`.
