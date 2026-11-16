- The public surface of `mime.assets` is smaller (SURFACE-REDUCTION's assets step, with the one `asset()`): 108 names
  that only MIME's own modules, its tests and its scripts used left its `__all__`, with no shim (the soft-tissue plugin
  imports `reference` and `resolve`, both kept); they are now internal. Public now: `asset`, `Asset`, `AssetRef`,
  `AssetRecord`, `Resolver`, `reference`, `resolve`, `check`, `CheckResult`, `Manifest`,
  `MANIFEST_FORMAT`, `read_manifest` and `Acceptance`; and, held public while the maintainer decides, the storage adapter's
  interface the lab author's guide teaches (`Adapter`, `Request`, `Credential`, `Opened`, `Redirect`, `AdapterError`,
  `NotFound`, `LoginRefused`, `Transient`, `register_adapter`) and `approved`, which the assets README and the script
  `mime modal` writes use. `tests/test_signature_types_are_public.py` now reads `mime.assets`; the types of names kept or
  deferred that no package exports wait as undecided. Out of `mime.assets`: `acceptances_path`, `acceptances_text`,
  `ACCEPTANCES_VARIABLE`, `active_pins`, `adapter_for`, `ADAPTER_GROUP`, `adapter_package`, `announce_downloads`,
  `Approval`, `approve`, `Asker`, `asking`, `ASSETS_DIR_VARIABLE`, `assets_of`, `AUDIT`, `audit`, `AuditReport`,
  `bake_bytes`, `BAKE_OUTPUT_FACTOR`, `beside`, `Builder`, `by_sha256`, `bytes_text`, `CA_BUNDLE_VARIABLE`, `cache_dir`,
  `Cached`, `can_ask`, `CHECK`, `CONFIRM_ABOVE_BYTES`, `CONFIRM_ABOVE_FRACTION`, `conflicts`, `container_of`,
  `credentials_from`, `derivation_by_hash`, `derivation_of`, `DerivationInput`, `DerivationRecord`, `derive`, `derived_terms`,
  `derived_verdicts`, `downloads_doing`, `fetch`, `FETCH`, `file_path`, `Finding`, `folder_assets`, `GROUP`, `is_cached`,
  `Json`, `Kind`, `KINDS`, `known`, `known_bytes`, `KnownBytes`, `Licence`, `LICENCES`, `licences_named`, `licences_of`,
  `licences_read`, `LIST`, `locate`, `location_problem`, `Locations`, `locations_path`, `LOCATIONS_VARIABLE`,
  `ManifestFile`, `mesh_format`, `mesh_path`, `named_file`, `named_path`, `offline`, `OFFLINE_VARIABLE`, `pinned`,
  `PlainAsker`, `Plan`, `probe`, `Probe`, `Produced`, `read_acceptances`, `read_derivation`, `read_locations`,
  `read_manifest_data`, `record_terms`, `registered_schemes`, `require_accepted`, `require_licences`,
  `RETRIEVE_LIMIT_BYTES`, `retrieve`, `revoke`, `shared_manifests`, `shipped_derivations`, `size_text`, `SOURCE_GROUP`,
  `SourceLocation`, `SOURCES`, `sources_text`, `sources_view`, `SourceView`, `stale_inputs`, `stated_licence`,
  `STOP_BELOW_BYTES`, `STOP_BELOW_FRACTION`, `stored_credential`, `tried_locations`, `under_test_prefix`,
  `watching_downloads`, `with_adapters`, `working_state`.
