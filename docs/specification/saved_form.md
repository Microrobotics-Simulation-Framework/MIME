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
  name, the input hash its lock records, the MIME version that saved it, and its DOI (the record it was pulled from,
  where it was pulled by a DOI, and otherwise the record its lock names, if any; a text of no DOI's shape is not
  recorded); the immediate one alone. It is part of the hashed input: a record edited after its lock was
  written is refused as `load.input-hash`, as any edit is. The new experiment's lock is its own and names no record;
  the folder it was made from is never written. A reader of the sixth form (that of 0.3.0.dev1) refuses a folder of the
  seventh or a later one as `load.newer-form`.
  *Cases:* `tests/saved/test_based_on.py::test_saving_writes_a_new_folder_whose_document_records_the_parent_and_whose_lock_is_its_own`,
  `tests/saved/test_based_on.py::test_the_record_names_the_parents_doi_where_its_lock_names_the_record_it_was_published_as`,
  `tests/saved/test_based_on.py::test_the_record_edited_by_hand_after_its_lock_was_written_is_refused`,
  `tests/saved/test_based_on.py::test_an_experiment_based_on_one_that_was_itself_based_on_another_records_its_immediate_parent`,
  `tests/saved/test_based_on.py::test_the_parents_folder_is_never_written`,
  `tests/saved/test_based_on.py::test_a_child_of_a_folder_the_released_wheel_saved_saves_in_the_current_form_loads_rebuilds_and_reproduces`,
  `tests/saved/test_based_on.py::test_the_record_names_the_mime_of_the_parents_lock_and_the_childs_lock_names_the_one_running`,
  `tests/saved/test_based_on.py::test_a_doi_that_is_not_a_dois_shape_is_not_recorded_and_the_warning_says_why`,
  `tests/sharing/test_based_on_shared.py::test_a_bundle_whose_lock_names_another_record_records_the_one_it_was_pulled_from_and_warns_naming_both`,
  `tests/saved/test_form7.py::test_dev1s_reader_refuses_a_folder_of_the_current_form_as_newer_each_file_once`.
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
- **[lock.previous-program]** A lock of a form before the one that added them that holds `reference.program`,
  `environment.usable_cpus` or `environment.blas_threads` is refused as `load.lock-shape`, never read as stating it; a lock of the seventh form
  without them reads as it did. A reader of the seventh form refuses a folder of the eighth as `load.newer-form`.
  *Cases:* `tests/saved/test_form8.py::test_an_earlier_lock_holding_a_key_version_eight_added_is_refused`,
  `tests/saved/test_form8.py::test_control_an_upgrade_that_accepts_the_keys_in_an_earlier_lock_is_caught_every_run`,
  `tests/saved/test_form8.py::test_an_earlier_reader_refuses_a_folder_of_version_eight_as_newer_each_file_once`,
  `tests/saved/test_form8.py::test_control_the_form_left_unstepped_is_read_by_version_sevens_reader_with_no_newer_form`.
- **[lock.reference-program]** The lock names the program the reference outputs were run by where it is not the
  history scan of whole control periods: `one-step`, every step its own compiled scan, which a save takes from a run
  chunked by one step. `verify` reruns the program the lock names on a graph built afresh, and the history scan for a
  lock that names none (every lock before the eighth form). A save takes the reference from the build's own graph and
  puts it back where its runs left it, so a run goes on from there bit for bit; from a graph built afresh where the
  build's graph was edited since the build.
  *Cases:* `tests/saved/test_reference_table.py::test_in_float32_on_a_first_build_the_saved_reference_is_what_a_fresh_graph_reruns`,
  `tests/saved/test_reference_table.py::test_in_float64_the_saved_reference_is_what_a_fresh_graph_reruns`,
  `tests/saved/test_save_reference.py::test_verify_builds_one_fresh_graph_and_reproduces_a_one_step_reference_bit_for_bit`,
  `tests/saved/test_save_reference.py::test_mime_verify_in_a_fresh_process_reproduces_a_one_step_reference_bit_for_bit`,
  `tests/saved/test_save_reference.py::test_control_the_lock_naming_the_other_program_fails_verify_where_the_two_differ`,
  `tests/saved/test_save_reference.py::test_saving_then_running_on_is_bit_for_bit_a_run_that_never_saved`,
  `tests/saved/test_save_reference.py::test_saving_after_a_step_count_the_observations_divider_does_not_divide_is_bit_for_bit_a_run_that_never_saved`,
  `tests/saved/test_save_reference.py::test_a_graph_edited_after_the_build_is_saved_from_a_fresh_graph`,
  `tests/saved/test_form8.py::test_a_version_seven_folder_loads_and_verifies_by_the_history_scan_and_its_environment_names_no_count`.
- **[lock.usable-cpus]** The lock's environment records how many CPUs the build's process could use (its affinity),
  by which XLA's CPU backend partitions a program's work: a load or `verify` under another count is in another
  environment, and reports it. A count either side could not read is never the same. A lock before the eighth form
  records none, and its environment is compared without one, as it was.
  *Cases:* `tests/saved/test_save_reference.py::test_a_folder_verified_at_the_count_it_was_saved_at_is_the_same_environment_bit_for_bit`,
  `tests/saved/test_save_reference.py::test_a_folder_verified_at_another_count_is_another_environment`,
  `tests/saved/test_save_reference.py::test_mime_verify_under_another_affinity_in_a_fresh_process_reports_another_environment`,
  `tests/saved/test_save_reference.py::test_a_count_either_side_could_not_read_is_never_the_same`,
  `tests/saved/test_form8.py::test_a_version_seven_folder_loads_and_verifies_by_the_history_scan_and_its_environment_names_no_count`.
- **[lock.blas-threads]** The lock's environment records how many threads the BLAS under JAX's CPU LAPACK kernels
  uses (the pool of the OpenBLAS SciPy ships, which `OPENBLAS_NUM_THREADS` sets, else the affinity), by which a routine
  that large splits its sums: a load or `verify` under another count is in another environment, and reports it. A
  count either side could not read is never the same. A lock before the eighth form records none, and its environment
  is compared without one, as it was.
  *Cases:* `tests/saved/test_save_reference.py::test_the_lock_records_the_blas_threads_and_a_folder_loaded_at_that_count_is_the_same_environment`,
  `tests/saved/test_save_reference.py::test_a_folder_loaded_at_another_blas_count_is_another_environment`,
  `tests/saved/test_save_reference.py::test_control_the_blas_count_dropped_from_the_comparison_fails_the_test_above`,
  `tests/saved/test_save_reference.py::test_the_mrs_case_saved_at_one_blas_count_reproduces_at_it_and_is_another_environment_at_another`,
  `tests/saved/test_save_reference.py::test_a_count_either_side_could_not_read_is_never_the_same`,
  `tests/saved/test_form8.py::test_a_version_seven_folder_loads_and_verifies_by_the_history_scan_and_its_environment_names_no_count`.
- **[lock.inertia-tensor]** The resolved record holds a body's `inertia` as its parts give it, the full tensor about
  its centre of mass in the body's frame, nine numbers row-major, from the ninth form; a declared inertia keeps the form
  it was declared in, three moments or nine numbers. A folder of an earlier form, whose record holds the parts' diagonal,
  loads as it was: its document names `rigid_inertial` 0.3.0 or 0.2.0, which builds its node from the diagonal, and the
  rebuilt record is compared in the folder's form, the parts' tensor written as its diagonal and a declaration's source
  naming the parts' tensor as the three numbers it named. A folder that rested on exploratory's waiver of a body's
  products of inertia is refused at load (`value.products-of-inertia`), by that version's words and naming the folder's
  route (re-create it from its script on the current version), never reproduced with them carried. A reader of the eighth form refuses a folder of the ninth as `load.newer-form`.
  *Cases:* `tests/saved/test_form9.py::test_version_nine_holds_the_parts_tensor_and_this_mime_writes_it`,
  `tests/saved/test_form9.py::test_a_form_eight_folder_with_inertia_from_parts_loads_on_the_kept_version_and_verifies_by_its_program`,
  `tests/saved/test_form9.py::test_control_with_the_write_back_removed_the_form_eight_folder_is_refused_naming_the_parts_inertia`,
  `tests/saved/test_form9.py::test_the_write_back_writes_the_parts_tensor_as_its_diagonal_and_a_source_as_the_three_it_named`,
  `tests/saved/test_form9.py::test_a_form_eight_folder_that_rested_on_the_waiver_of_its_products_is_refused_by_the_kept_versions_words`,
  `tests/saved/test_form9.py::test_a_form_seven_based_on_child_loads_keeps_its_parent_and_verifies`,
  `tests/saved/test_form9.py::test_a_folder_holding_a_full_inertia_tensor_round_trips_in_a_fresh_process`,
  `tests/saved/test_form9.py::test_version_eights_reader_refuses_a_folder_of_version_nine_as_newer_each_file_once`,
  `tests/saved/test_form9.py::test_control_the_form_left_unstepped_is_read_by_version_eights_reader_with_no_newer_form`.
- **[lock.blocks]** From the tenth form, where `verify` claims no bits, every saved field is split into **blocks by
  unit**, read from the units MIME declares for it (its node's symbol table, else its node's typed state, where a
  field's units stand beside its type as `Units`, then the edges that read it, the stage's binding, the observations'
  outputs, a flag's): a composite's units list its parts, separated by semicolons, its leading parts vectors of three,
  so a pose is its position and its quaternion, a twist its velocity and its angular velocity, a wrench its force and
  its torque, and a resistance its translational, coupling and rotational blocks. A field of one unit is one block, and
  so is one whose units are not declared; every field the saved cases' builds hold declares its units. A block is within
  tolerance when at every reference step its largest difference is at most `max(rtol` times its largest magnitude over
  the reference, `resolution_factor` times its `resolution``)` plus `atol`, written as zero. The **resolution** is
  measured at save: the reference steps run once more, by the same compiled program, from a start one unit in the last
  place off (every nonzero floating number of the initial state, and of the small parameters the compiled step reads
  (`NUDGED_SIZE`), one ulp up), and the lock records each block's largest divergence (`reference.resolution`, `null`
  where it is not finite) and the factor (`reference.resolution_factor`), the two keys the form adds. A resolution that
  is zero or not finite gives no floor. The start moves the parameters as well as the state, on purpose: on the tube
  demo a start that moved the state alone measured every block's divergence zero, in both precisions, since a body
  released at rest at the origin starts from zeros. Where a parameter it moves is one an MRS drag holds a value made
  from, the rerun makes that value once, from the moved parameters, and the compiled step keeps it there rather than
  making it at each step, with the resolution unchanged (measured in `mime.saved._scales`). Held points
  (`HeldPoints`) are judged only on the rows whose paired force (`HeldForces`) acts in the reference or the rerun,
  their magnitude and resolution too. A field that is not of floating point is compared
  exactly. A folder of the ninth form or earlier is judged as its form says, element by element
  (`|a - b| <= atol + rtol |b|` with its own numbers), in `verify`, `verify --run` and the load's comparison of the
  derived graph, and one that holds either new key is refused. A reader of the ninth form refuses a folder of the tenth
  as `load.newer-form`.
  *Cases:* `tests/saved/test_form10.py::test_version_ten_writes_its_precisions_tolerance_with_no_absolute_part_and_each_fields_resolution`,
  `tests/saved/test_form10.py::test_a_form_nine_folder_loads_with_the_tolerance_it_was_written_with_and_verifies_element_by_element`,
  `tests/saved/test_form10.py::test_a_form_nine_folder_is_judged_element_by_element_and_the_same_rerun_as_form_ten_block_by_block`,
  `tests/saved/test_form10.py::test_a_form_nine_lock_holding_a_resolution_is_refused_and_the_same_lock_without_it_loads`,
  `tests/saved/test_form10.py::test_version_nines_reader_refuses_a_folder_of_version_ten_as_newer_each_file_once`,
  `tests/saved/test_form10.py::test_control_the_form_left_unstepped_is_read_by_version_nines_reader_with_no_newer_form`,
  `tests/saved/test_verify_blocks.py::test_a_pose_twist_and_wrench_split_into_their_first_three_and_the_rest`,
  `tests/saved/test_verify_blocks.py::test_a_resistance_splits_into_its_translational_coupling_and_rotational_blocks`,
  `tests/saved/test_verify_blocks.py::test_one_unit_is_one_block_and_so_is_a_field_whose_units_are_not_declared_or_do_not_fit`,
  `tests/saved/test_verify_blocks.py::test_every_composite_interface_of_the_vocabulary_splits_into_one_block_per_unit`,
  `tests/saved/test_verify_blocks.py::test_the_sliding_balls_fields_take_their_units_from_its_build_and_its_lock_records_each_blocks_resolution`,
  `tests/saved/test_verify_blocks.py::test_m1_the_balls_position_moved_8e_6_m_is_moved_at_every_step`,
  `tests/saved/test_verify_blocks.py::test_m2_the_balls_angular_velocity_zeroed_flipped_or_doubled_is_moved`,
  `tests/saved/test_verify_blocks.py::test_m3_a_resistances_rotational_or_coupling_block_times_100_is_moved`,
  `tests/saved/test_verify_blocks.py::test_a_torque_where_the_reference_holds_none_is_moved`,
  `tests/saved/test_verify_blocks.py::test_a_plant_just_beyond_a_blocks_bound_is_caught_at_every_step_with_either_sign`,
  `tests/saved/test_verify_blocks.py::test_a_resolution_that_is_zero_or_not_a_finite_positive_number_gives_no_floor`,
  `tests/saved/test_verify_blocks.py::test_a_held_point_of_a_row_that_takes_a_force_moved_at_ten_times_its_bound_is_moved_at_every_step`,
  `tests/saved/test_verify_blocks.py::test_a_held_point_of_a_row_that_takes_no_force_jumping_two_millimetres_is_not_reported`,
  `tests/saved/test_save_reference.py::test_a_save_after_a_run_chunked_by_one_step_builds_and_compiles_nothing`,
  `tests/saved/test_held_once.py::test_the_nudged_pass_makes_the_drags_matrix_once_and_the_step_none`,
  `tests/saved/test_held_once.py::test_the_resolution_and_reference_are_the_steps_that_make_the_matrix_at_every_step_bit_for_bit`,
  `tests/saved/test_held_once.py::test_control_a_nudged_pass_given_a_matrix_made_from_the_unnudged_leaves_records_another_resolution`,
  `tests/unit/test_field_kinds.py::test_a_new_nodes_held_points_pair_with_its_one_field_of_held_forces`,
  `tests/unit/test_field_kinds.py::test_control_a_typed_state_that_does_not_name_exactly_the_fields_types_none_of_them`,
  `tests/unit/test_field_kinds.py::test_a_new_nodes_typed_state_declares_the_units_of_its_fields_beside_their_types`,
  `tests/unit/test_field_kinds.py::test_control_units_two_typed_states_declare_otherwise_or_none_names_the_fields_leave_them_undeclared`,
  `tests/saved/test_declared_units.py::test_every_field_a_save_writes_has_declared_units`,
  `tests/saved/test_declared_units.py::test_control_without_the_typed_states_declarations_the_kept_bodys_twist_and_the_drags_diagnostics_are_reported`,
  `tests/saved/test_declared_units.py::test_control_a_typed_state_that_disagrees_with_a_symbol_row_is_reported`.
- **[lock.tolerance]** A save of the tenth form writes its precision's `rtol` (`FLOAT32_RTOL` or `FLOAT64_RTOL` of
  `mime.saved._scales`) and the floor's factor (`RESOLUTION_FACTOR`), and a folder keeps the ones it was written with.
  Each is pinned in a table of the measurements it was set from, with where each was made and its margin
  (`tests/saved/test_tolerance_ratchet.py`): in float32 the spreads between runs on one T4 GPU and against its CPU,
  and across three counts of usable CPUs of one laptop, on one demo over eight steps; in float64 across CPU counts
  only, no GPU measured; the factor against the demo's one-ulp resolution, and below the angular velocity's ceiling. The
  measurements the repository holds are computed again from it. A constant is raised only by a new measurement and a
  design decision entered in that table; lowering one needs neither. The tolerance governs how another environment's
  rounding, or a non-deterministic GPU's, is judged on runs like the one measured; it states nothing of accuracy, and is
  no promise that no other machine or experiment reads moved.
  *Cases:* `tests/saved/test_tolerance_ratchet.py::test_each_constant_is_at_most_its_pin`,
  `tests/saved/test_tolerance_ratchet.py::test_control_each_constant_raised_fails_and_lowered_passes`,
  `tests/saved/test_tolerance_ratchet.py::test_each_pin_stands_its_margin_above_its_largest_measurement`,
  `tests/saved/test_tolerance_ratchet.py::test_the_measurements_the_repository_holds_are_what_its_artefacts_give`,
  `tests/saved/test_tolerance_ratchet.py::test_k_stands_below_the_ceiling_the_angular_velocity_needs_by_its_margin`,
  `tests/saved/test_tolerance_ratchet.py::test_a_save_writes_the_pinned_constants_into_its_lock`,
  `tests/saved/test_form10.py::test_a_folder_keeps_the_tolerance_it_was_written_with`,
  `tests/saved/test_verify_blocks.py::test_across_cpu_counts_every_block_is_within_its_bound`,
  `tests/saved/test_verify_blocks.py::test_control_without_its_resolution_the_angular_velocity_across_counts_is_moved`,
  `tests/saved/test_verify_blocks.py::test_no_false_moved_the_t4_spread_planted_per_block_verifies_within_tolerance_with_the_warning`,
  `tests/saved/test_verify_blocks.py::test_no_false_moved_is_not_on_the_edge_the_t4_spread_sits_well_inside_each_floor`,
  `tests/saved/test_verify_blocks.py::test_today_the_same_plant_is_moved`.
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
  development release is refused by the next, whatever else is the same. The refusal is its own, `load.other-release`,
  the first thing `Experiment.load` does after reading the folder, before any entry resolves or anything builds, so a
  folder whose document no longer resolves under this MIME (an entry version it no longer registers, a reference it
  no longer builds) is still refused for the MIME that saved it, not for an entry to install: it names both versions ("saved by MIME 0.3.0.dev1, and this is MIME
  0.3.0.dev2"), says that development releases do not load each other's folders, and says what to do: run the script
  that built the experiment under this MIME and save it again (`built.save(folder)`), or load the folder as it is under
  the MIME that saved it (`pip install mime-engine==<that version>`). A folder unpacked from a published record
  (`mime pull`: its lock records the publication) also says so and names the record's page. `load.input-hash` keeps
  its meaning, a document changed after its lock was written, and is raised only where the versions agree. It holds of
  a saved folder in every use of it (`Experiment.load`, `mime verify`, `mime run`, `mime pack`); a bundle unpacks
  (`mime pull` names the MIME that packed it) but its folder is refused in the same way; and a run folder's outputs
  still read (`RunResult.read`) while the `experiment/` folder inside it is a saved folder, so it is refused. The
  compatibility policy for releases after the development ones is designed before the first users; this is not it, and
  two final releases (0.3.0 and 0.3.1) still meet `load.input-hash` and its words about a changed document. Versions
  are compared as whole strings, local part included (a checkout's own folders load; another commit's, or a dirty
  tree's against a clean one, are refused), and a development release is judged by `packaging.version`.
  *Cases:* `tests/saved/test_other_release.py::test_a_lock_recording_another_development_release_is_refused_by_its_own_refusal`,
  `tests/saved/test_other_release.py::test_a_folder_from_another_release_whose_document_no_longer_resolves_is_still_refused_as_another_release`,
  `tests/saved/test_other_release.py::test_a_version_with_a_local_part_is_judged_on_the_whole_string`,
  `tests/saved/test_load.py::test_a_folder_saved_by_another_mime_version_is_refused_and_one_saved_by_this_one_loads`.
- **[rebuild.edited-after-load]** An experiment that was loaded and then edited (a body added in a notebook cell run
  after the load) is refused at its next build as `load.input-hash`, in words about the edit: "edited after it was
  loaded", naming what changed and, in a notebook or IPython, saying that a line or cell run after the load changed
  it. A loaded experiment is checked against its lock at every build, and a save builds first, so the edited experiment
  can be neither built nor saved. The action is to load the folder again, which drops the edit, or to keep the edit in
  a new experiment made from the folder, `Experiment.based_on(<folder>, name=...)`, and save that.
  *Cases:* `tests/saved/test_other_release.py::test_an_experiment_edited_after_it_was_loaded_is_refused_in_words_about_the_edit`,
  `tests/saved/test_other_release.py::test_the_actions_the_edited_refusal_names_work_and_the_one_it_rules_out_is_refused`,
  `tests/saved/test_other_release.py::test_the_action_names_based_on_now_that_it_exists`.
- **[rebuild.stage]** A stage written by a stage writer this MIME does not have is refused as `load.stage-writer`; in
  the environment the lock was written in, a rebuilt stage of other bytes is refused as `load.stage-hash`.
  *Cases:* `tests/saved/test_stage_writer.py::test_the_control_a_lock_naming_another_writer_or_a_stage_neither_writes_is_refused`.
- **[rebuild.derived]** In the environment the lock was written in, the rebuilt graph must be the saved one, hash
  for hash, or the load is refused as `load.derived-hash`. In another environment its physics is compared within the
  lock's tolerance, and each number beyond it, and each static array that differs, is reported (the observations are
  not compared); there a difference in the derived graph or in the rebuilt stage is reported, never refused. From the
  tenth form each list of numbers and each static array is judged against its own largest magnitude, and a single
  number against its own; before it, element by element.
  *Cases:* `tests/saved/test_load.py::test_a_factory_that_builds_another_graph_under_the_same_version_is_refused`,
  `tests/saved/test_load.py::test_in_another_environment_differences_are_reported_and_outputs_compared_within_tolerance`,
  `tests/saved/test_form10.py::test_from_form_ten_the_derived_graphs_numbers_are_judged_on_their_own_largest_magnitude`.
- **[rebuild.same]** A folder saved and loaded in one environment rebuilds the same experiment, and `mime verify`
  reruns it to the reference outputs bit for bit, unless its reference was made on a GPU with XLA's deterministic mode
  off: two such runs of one program differ in their last bits, so it is judged within the lock's tolerance everywhere,
  the GPU it was made on included (block by block from the tenth form; a folder of an earlier form keeps its
  element-by-element rule, under which such a folder can still read moved where it was made). Made in deterministic mode, it verifies bit for bit in the same environment again. In another environment every folder is judged within its
  tolerance. `mime verify --run` claims the same of a run folder, and the card's tolerance line states the same levels.
  *Cases:* `tests/saved/test_load.py::test_a_saved_experiment_loads_and_verifies_bit_for_bit`,
  `tests/saved/test_verify_blocks.py::test_a_deterministic_folder_verified_in_the_same_environment_is_bit_for_bit`,
  `tests/saved/test_verify_blocks.py::test_the_sliding_ball_verifies_bit_for_bit_where_its_lock_was_written`,
  `tests/saved/test_verify_blocks.py::test_a_gpu_reference_made_with_deterministic_mode_off_reruns_within_tolerance_with_the_warning`,
  `tests/saved/test_deterministic_warning.py::test_verify_of_a_cpu_or_a_deterministic_reference_never_warns`,
  `tests/saved/test_deterministic_warning.py::test_the_cards_tolerance_line_states_each_level_and_the_rule_of_the_folders_form`.
- **[rebuild.deterministic]** A reference written on a GPU with deterministic mode off carries one warning where the
  save writes it, and the same line ends `verify`'s report of its folder: deterministic mode is off, so the reference
  verifies within tolerance, not bit for bit; for bit-for-bit reproducibility, rerun with `--deterministic`. Neither
  is said on the CPU, nor of a reference made in deterministic mode. `--deterministic` on `mime run` and `mime verify`
  turns the mode on before JAX's backend starts, since XLA reads its flags only then (`mime modal --deterministic`
  sets it in the script it writes); from Python,
  `mime.env.deterministic_mode()` does the same before anything computes, and is refused as `env.deterministic-late`
  once the backend has started, unless the flags already turn the mode on or an earlier call set them itself, which a
  second call answers as it did. On the CPU it changes nothing, says so in one line, and the lock records the flags as
  they were.
  *Cases:* `tests/saved/test_deterministic_warning.py::test_a_save_on_a_gpu_with_deterministic_mode_off_warns_once`,
  `tests/saved/test_deterministic_warning.py::test_a_save_on_the_cpu_never_warns`,
  `tests/saved/test_deterministic_warning.py::test_a_save_on_a_gpu_in_deterministic_mode_never_warns`,
  `tests/saved/test_deterministic_warning.py::test_verify_of_a_gpu_reference_made_with_deterministic_mode_off_ends_with_the_warning`,
  `tests/saved/test_deterministic_warning.py::test_verify_of_a_cpu_or_a_deterministic_reference_never_warns`,
  `tests/env/test_deterministic_mode.py::test_xla_reads_its_flags_once_when_the_backend_starts_and_importing_mime_starts_none`,
  `tests/env/test_deterministic_mode.py::test_after_the_backend_started_the_python_route_is_refused_naming_the_routes_that_work`,
  `tests/env/test_deterministic_mode.py::test_with_the_mode_already_in_the_flags_it_is_not_refused_and_on_the_cpu_it_is_not_on`,
  `tests/env/test_deterministic_mode.py::test_in_a_fresh_cpu_process_it_changes_nothing_and_the_flags_are_as_they_were`,
  `tests/env/test_deterministic_mode.py::test_in_a_fresh_cpu_process_a_second_call_answers_as_the_first_and_is_not_refused`,
  `tests/env/test_deterministic_mode.py::test_mime_verify_deterministic_on_the_cpu_says_it_changes_nothing_and_verifies_as_without_it`,
  `tests/env/test_deterministic_mode.py::test_mime_run_deterministic_on_the_cpu_records_the_flags_as_they_were`.

## The schemas

- **[schema.second-reader]** The JSON Schemas are generated from the same closed TypedDicts the loader checks
  against. Over the committed folders, the planted inputs and the files drawn from the TypedDicts, a schema and the
  loader disagree exactly on the gaps the page's table records; where they disagree, the loader decides.
  *Cases:* `tests/saved/test_json_schema.py::test_the_readers_disagree_exactly_on_the_recorded_gaps`,
  `tests/saved/test_json_schema.py::test_on_drawn_files_and_their_changes_the_readers_agree`.
- **[schema.golden]** Every committed document and lock validates against its schema, as the loader reads it.
  *Cases:* `tests/saved/test_json_schema.py::test_every_golden_document_and_lock_validates_as_the_loader_reads_it`.
