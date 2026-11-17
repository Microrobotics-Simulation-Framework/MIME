"""A controller inside the compiled step (its first mode): the node that runs its law's step at the controller's own
rate, and the delay line that holds its writes for its latency before they reach their commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast, override

from jax import Array
from jaxtyping import Float
from maddening.core.node import BoundaryInputSpec
from typing_extensions import ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, jnp, node_fields, node_input, typed_view

#: A controller's step, ``(state, reads) -> (state, writes)``, each a mapping of arrays by label.
type Step = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]]
#: A read label's units, as the vocabulary spells them, and its shape.
type ReadSpec = tuple[str, tuple[int, ...]]
#: The prefix of a controller node's leaf holding one of its law's settings (``setting_gain``).
SETTING = "setting_"
#: The prefix of a controller node's leaf holding a number of a believed model (``ctx.believed``): its key and its
#: parameter joined by double underscores, the key's dots as double underscores and its colon (an observation's id,
#: ``camera:``) as ``_c_`` (``believed_coils__magnetic__coefficients``, ``believed_camera_c___fx``).
BELIEVED = "believed_"
#: The prefix of the name a law is made again from for a believed number: ``believed:<key>:<parameter>``.
BELIEVED_NAME = "believed:"


def believed_name(key: str, parameter: str) -> str:
    """The name a belief's number is made again under: ``believed:<key>:<parameter>``."""
    return f"{BELIEVED_NAME}{key}:{parameter}"


def split_believed_name(name: str) -> tuple[str, str]:
    """``(key, parameter)`` of ``believed:<key>:<parameter>``, the inverse of :func:`believed_name`: a key may hold a
    colon (an observation's id, ``camera:``), a parameter never does, so the split is at the last."""
    key, _, parameter = name[len(BELIEVED_NAME):].rpartition(":")
    return key, parameter


#: The prefix of a controller node's leaf holding a number of a command port its law reads (``ctx.allocator``'s bounds
#: and slew): the apparatus, the command and the number joined by double underscores (``port_coils__currents__high``).
PORT = "port_"
#: The prefix of the name a law is made again from for a port's number: ``port:<apparatus>.<command>:<number>``.
PORT_NAME = "port:"


def port_name(apparatus: str, command: str, number: str) -> str:
    """The name a command port's number (``low``, ``high``, ``slew``) is made again under, as the controller reads it:
    ``port:<apparatus>.<command>:<number>``."""
    return f"{PORT_NAME}{apparatus}.{command}:{number}"


def leaf_param(name: str) -> str:
    """The controller node's leaf a name its law is made again from is held as: ``setting_<name>`` for a setting,
    ``believed_<key>__<parameter>`` for a believed number (``believed:<key>:<parameter>``, the key with its dots and its
    colon written out, which a leaf name holds none of), ``port_<apparatus>__<command>__<number>`` for a port's number
    (``port:<apparatus>.<command>:<number>``, the command's dots written out)."""
    if name.startswith(BELIEVED_NAME):
        key, parameter = split_believed_name(name)
        return f"{BELIEVED}{key.replace('.', '__').replace(':', '_c_')}__{parameter}"
    if name.startswith(PORT_NAME):
        key, _, number = name[len(PORT_NAME):].rpartition(":")
        return f"{PORT}{key.replace('.', '__')}__{number}"
    return f"{SETTING}{name}"


class ControllerState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``state_<key>`` for each key of the law's state, and ``write_<label>`` for each label it writes."""


class ControllerInputs(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``read_<label>`` for each label it reads."""


class DelayState(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``<label>``, the writes acting this step, and ``waiting_<label>``, those taken since, oldest first."""


class DelayInputs(TypedDict, extra_items=ReadOnly[Float[Array, "..."]]):
    """``<label>``, the controller's writes this step."""


class ControllerNode(MimeNode):
    """A controller's law inside the compiled step. At each of its ticks it reads ``read_<label>`` for each label it
    reads, steps its law, and holds the law's state as ``state_<key>`` and its writes as ``write_<label>``, each write
    in the dtype of the command it feeds. It ticks every ``timestep``, a whole number of base steps: between ticks
    MADDENING's multi-rate schedule holds its state, so its writes stay in force.

    Its law is the controller's, made from the experiment's declaration of it, which the saved form holds: ``step`` is
    called inside the compiled step and is no parameter, so this node, unlike other MIME nodes, is not rebuilt from its
    ``params`` alone. Those name the controller and hold its reads' units and shapes, its law's state at the start
    (``initial_state``) and the writes it holds before its first (``initial_writes``), which the derived hash
    covers."""

    #: Its start and writes are its law's, made at build time in a dtype of the law's own (the arm tracking keeps float32
    #: in a float64 run): taken as given, narrowed in a float32 run.
    REFUSES_NARROW_ARGUMENTS = False

    def __init__(self, name: str, timestep: float, *, step: Step, controller: str, reads: Mapping[str, ReadSpec],
                 initial_state: Mapping[str, Any], initial_writes: Mapping[str, Any],
                 settings: Mapping[str, Any] | None = None,
                 remake: Callable[[Mapping[str, Any]], Step] | None = None) -> None:
        held = {leaf_param(name_): jnp.asarray(value) for name_, value in sorted((settings or {}).items())}
        super().__init__(name, timestep, controller=controller,
                         reads={label: (units, tuple(shape)) for label, (units, shape) in sorted(reads.items())},
                         initial_state={key: jnp.asarray(value) for key, value in sorted(initial_state.items())},
                         initial_writes={label: jnp.asarray(value) for label, value in sorted(initial_writes.items())},
                         **held)
        self.law_step = step
        #: Where settings are leaves (``settings``, each ``setting_<name>``), the law's step made again from them
        #: inside the traced step: ``remake(settings) -> step``.
        self.remake = remake
        self.leaf_settings = tuple(sorted(settings or {}))

    def step_at(self, p: Mapping[str, Any]) -> Step:
        """The law's step at the step constants ``p``: made again from the leaves it holds, or the built one."""
        if self.remake is None:
            return self.law_step
        return self.remake({name: p[leaf_param(name)] for name in self.leaf_settings})

    @override
    def to_dict(self) -> dict[str, Any]:
        """MADDENING's description of the node, its ``params``. Its step is saved another way: it is the controller's
        law, which the experiment's saved form holds and the build makes again, so a graph rebuilt from this
        description alone has no step for this node."""
        held: Any = super()
        return cast(dict[str, Any], held.to_dict())

    def _arrays(self, key: str) -> Mapping[str, Array]:
        return cast(Mapping[str, Array], self.constants()[key])

    def _reads(self) -> Mapping[str, ReadSpec]:
        return cast(Mapping[str, ReadSpec], self.constants()["reads"])

    @override
    def initial_state(self) -> NodeFields:
        return node_fields({**{f"state_{key}": value for key, value in self._arrays("initial_state").items()},
                            **{f"write_{label}": value for label, value in self._arrays("initial_writes").items()}})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {f"read_{label}": node_input(units, shape) for label, (units, shape) in self._reads().items()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(ControllerState, state), typed_view(ControllerInputs, boundary_inputs)
        keys, writes = self._arrays("initial_state"), self._arrays("initial_writes")
        after, written = self.step_at(p)({key: held[f"state_{key}"] for key in keys},
                                         {label: given[f"read_{label}"] for label in self._reads()})
        return node_fields({**{f"state_{key}": after[key] for key in keys},
                            **{f"write_{label}": jnp.asarray(written[label], dtype=writes[label].dtype)
                               for label in writes}})


class DelayNode(MimeNode):
    """A delay line of ``latency`` base steps between a controller and the commands it writes: each step it takes the
    controller's writes (``<label>``), passes on as its state ``<label>`` those it took ``latency`` steps before, and
    keeps the ones between waiting (``waiting_<label>``, oldest first). It starts from the writes the controller holds
    before its first, ``initial_writes``, as if taken on each of the ``latency`` steps before the run."""

    def __init__(self, name: str, timestep: float, *, latency: int, units: Mapping[str, str],
                 initial_writes: Mapping[str, Any]) -> None:
        if latency < 1:
            raise ValueError(f"a delay line holds at least one step, not {latency}")
        super().__init__(name, timestep, latency=latency, units=dict(sorted(units.items())),
                         initial_writes={label: jnp.asarray(value) for label, value in sorted(initial_writes.items())})

    def _writes(self) -> Mapping[str, Array]:
        return cast(Mapping[str, Array], self.constants()["initial_writes"])

    @override
    def initial_state(self) -> NodeFields:
        latency = cast(int, self.constants()["latency"])
        return node_fields({**dict(self._writes()),
                            **{f"waiting_{label}": jnp.stack([value] * latency)
                               for label, value in self._writes().items()}})

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        units = cast(Mapping[str, str], self.constants()["units"])
        return {label: node_input(units[label], tuple(value.shape)) for label, value in self._writes().items()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(DelayState, state), typed_view(DelayInputs, boundary_inputs)
        out: dict[str, Array] = {}
        for label in self._writes():
            waiting = cast(Array, held[f"waiting_{label}"])
            out[label] = waiting[0]
            out[f"waiting_{label}"] = jnp.concatenate([waiting[1:], given[label][None]])
        return node_fields(out)


__all__ = ["BELIEVED", "BELIEVED_NAME", "PORT", "PORT_NAME", "SETTING", "ControllerInputs", "ControllerNode", "ControllerState",
           "DelayInputs", "DelayNode", "DelayState", "ReadSpec", "Step", "believed_name", "leaf_param", "port_name",
           "split_believed_name"]
