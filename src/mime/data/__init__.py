"""Package data shipped in the wheel: small files, each a record of ``assets/packaged.json`` pinned by its sha256
(surrogate weights, the exact sphere-plane resistance table, copies of OpenMaterica's example records, three
electromagnetic navigation systems' calibrations), the records of MIME's remote assets (``assets/``), derivation
records, the environment's pins and a visual set's manifest.

An experiment names a packaged file as it names any record, ``asset("dejongh_mlp_v2")`` (:func:`mime.assets.asset`),
and reads its bytes in place, held to the sha256 the experiment is written against. Nothing here is code."""

__all__: list[str] = []
