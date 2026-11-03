# Registrations

A registration is how a law asks a dynamics to **include another thing in its own update**: a flow solver that holds an
immersed body inside its fluid step is the case the kind exists for. The dynamics reads the registered thing's state and
delivers back into its targets, all inside its own node. So a registration is not delivered by an edge: **the receiving
dynamics takes it when it is constructed.**

**What happens between the nodes.**
1. A law plans a registration: the role of the dynamics that includes the thing, the role of the thing it includes (the
   subject), the registration's kind, the target, the state it reads of the subject (by label and interface) and the
   targets it delivers back into (by label).
2. The target must be a registration target in the vocabulary. The dynamics bound to the receiving role must accept
   both the target and the registration's kind, as its capability record declares; one that does not is refused
   (`registration.not-accepted`), naming the dynamics that accept that kind.
3. The subject's reads are resolved as state reads of the subject, and its returns as deliveries into the subject's
   own targets, with the checks a delivery has anywhere (the subject's dynamics must accept the target it is returned
   into). Any that fails to resolve stops the registration; none is half built.
4. The resolved registration is handed to the dynamics at construction with the others it holds, and the dynamics' node
   carries them. The dynamics' page says what it does with a registered thing, and how its result returns.
5. A post-build check refuses an edge into a target of this kind (`post-build.not-by-edge`), as it does for an
   inequality constraint. No test names this code, so this step is read from the source and no test shows it.

**What this page does not say.** What the including dynamics computes for the thing (the coupling between a fluid and an
immersed body, say) is that dynamics' node page, and what is established about it is its claims'. The build only checks
that the pieces fit: that the dynamics takes the kind, and that what it reads and returns exists.
