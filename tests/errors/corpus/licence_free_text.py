"""Terms and tiers: an experiment's licence written as free text, which the saved form does not hold."""
# corpus: codes terms.licence
import tempfile

from mime.testing.toys import full_toy

exp = full_toy()
exp.licence("Creative Commons Attribution")
exp.save(tempfile.mkdtemp())  # corpus: here
