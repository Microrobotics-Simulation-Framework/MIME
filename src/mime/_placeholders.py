"""What ``mime scaffold --yes`` writes where a licence or a namespace is wanted and neither ``--set`` nor the person's
defaults gave one: a marker, never a choice (a licence is the person's, and a real-looking namespace is a name in the
registry). Every check that reads a licence, a block's id or a plugin's namespace refuses it, naming this."""

PLACEHOLDER_LICENCE = "LicenseRef-CHOOSE-A-LICENCE"
PLACEHOLDER_NAMESPACE = "choose_a_namespace"
