"""Standard feedback policies for microrobot control.

StepOutDetector: monitors phase_error from PhaseTrackingNode and drops
the actuation frequency when step-out is detected.
"""

from __future__ import annotations

from mime.control.policy import ControlPolicy, ExternalInputs, PolicyState


class StepOutDetector(ControlPolicy):
    """Feedback policy that detects step-out and drops frequency.

    Monitors the phase_error from PhaseTrackingNode. When phase_error
    exceeds the threshold (pi/2 by default), the policy drops the
    actuation frequency to recovery_frequency and holds it for
    recovery_duration seconds before ramping back up.

    Parameters
    ----------
    phase_node : str
        Name of the PhaseTrackingNode to monitor.
    field_node : str
        Name of the ExternalMagneticFieldNode to command.
    nominal_frequency : float
        Normal operating frequency [Hz].
    recovery_frequency : float
        Frequency to drop to on step-out [Hz].
    recovery_duration : float
        Time to hold at recovery frequency [s].
    ramp_rate : float
        Rate of frequency increase during recovery ramp [Hz/s].
    phase_threshold : float
        Phase error threshold for step-out detection [rad].
        Default: pi/2.
    field_strength_mt : float
        Constant field strength [mT].
    """

    def __init__(
        self,
        phase_node: str = "phase",
        field_node: str = "external_field",
        nominal_frequency: float = 20.0,
        recovery_frequency: float = 5.0,
        recovery_duration: float = 1.0,
        ramp_rate: float = 10.0,
        phase_threshold: float = 1.5708,  # pi/2
        field_strength_mt: float = 10.0,
    ):
        self.phase_node = phase_node
        self.field_node = field_node
        self.nominal_frequency = nominal_frequency
        self.recovery_frequency = recovery_frequency
        self.recovery_duration = recovery_duration
        self.ramp_rate = ramp_rate
        self.phase_threshold = phase_threshold
        self.field_strength_mt = field_strength_mt

    def __call__(
        self,
        t: float,
        observed_state: dict,
        policy_state: PolicyState,
    ) -> tuple[ExternalInputs, PolicyState]:
        mode = policy_state.get("mode", "nominal")
        recovery_start = policy_state.get("recovery_start", 0.0)
        current_freq = policy_state.get("current_freq", self.nominal_frequency)

        # Read phase error from observed state
        phase_data = observed_state.get(self.phase_node, {})
        phase_error = phase_data.get("phase_error", 0.0)

        # Convert JAX scalar to Python float for comparisons
        if hasattr(phase_error, 'item'):
            phase_error = float(phase_error)

        if mode == "nominal":
            if phase_error > self.phase_threshold:
                # Step-out detected — switch to recovery
                mode = "recovery"
                recovery_start = t
                current_freq = self.recovery_frequency
            else:
                current_freq = self.nominal_frequency

        elif mode == "recovery":
            elapsed = t - recovery_start
            if elapsed < self.recovery_duration:
                # Hold at recovery frequency
                current_freq = self.recovery_frequency
            else:
                # Ramp back up
                ramp_elapsed = elapsed - self.recovery_duration
                current_freq = min(
                    self.recovery_frequency + self.ramp_rate * ramp_elapsed,
                    self.nominal_frequency,
                )
                if current_freq >= self.nominal_frequency:
                    mode = "nominal"
                    current_freq = self.nominal_frequency

        ext = {
            self.field_node: {
                "frequency_hz": current_freq,
                "field_strength_mt": self.field_strength_mt,
            }
        }
        new_ps = {
            "mode": mode,
            "recovery_start": recovery_start,
            "current_freq": current_freq,
        }
        return ext, new_ps

    def initial_policy_state(self) -> PolicyState:
        return {
            "mode": "nominal",
            "recovery_start": 0.0,
            "current_freq": self.nominal_frequency,
        }
