- **The test suite can no longer write a developer's real configuration.** A test that saved defaults (`mime.push._defaults.write_defaults`) or a
  credential without a home of its own wrote under the shell's `~/.config/mime/`, which once overwrote a maintainer's real `defaults.toml`. An
  autouse fixture in `tests/conftest.py` now gives every test its own config and state directories (`XDG_CONFIG_HOME`, `XDG_STATE_HOME`, and
  Windows' application-data overrides); `HOME` and the XLA compilation cache are left alone, and a test that sets its own home still replaces these.
  `tests/test_config_home_isolation.py` writes defaults and a credential through the public functions against a sentinel home and fails without the
  fixture. A test change; MIME is unchanged.
