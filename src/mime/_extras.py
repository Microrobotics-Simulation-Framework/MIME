"""The message an optional extra's absence raises, held where a module that needs no extra can read it."""

#: What importing ``mime.testing`` raises without the ``verify`` extra (Hypothesis, through MADDENING's).
VERIFY_EXTRA_MISSING = "mime.testing needs the verify extra: pip install 'mime-engine[verify]'"
