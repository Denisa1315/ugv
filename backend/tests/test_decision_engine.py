from app.modules.decision.config import DecisionThresholds
from app.modules.decision.engine import DecisionEngine
from app.modules.decision.schemas import ActionCommand, SystemState


def make_engine(**overrides):
    return DecisionEngine(DecisionThresholds(**overrides))


# -- "no obstacle detected" must never be treated as safe when confidence is low --


def test_low_confidence_never_yields_safe_even_with_zero_risk():
    engine = make_engine()
    result = engine.evaluate(
        perception_confidence=0.05,  # e.g. a blurry/degenerate frame, zero detections
        depth_confidence=0.9,
        localization_confidence=0.9,
        local_risk=0.0,  # nothing flagged as risky — must NOT be read as "safe"
        goal_reachable=True,
        raw_min_clearance_m=None,
    )
    assert result.state != SystemState.SAFE
    assert result.state == SystemState.CRITICAL  # combined_confidence (min) is 0.05, below degraded_confidence_min
    assert result.action == ActionCommand.PAUSE


def test_combined_confidence_is_the_weakest_sensor_not_an_average():
    engine = make_engine()
    result = engine.evaluate(
        perception_confidence=0.95,
        depth_confidence=0.95,
        localization_confidence=0.1,  # one bad sensor
        local_risk=0.0,
        goal_reachable=True,
        raw_min_clearance_m=None,
    )
    assert result.combined_confidence == 0.1
    assert result.state != SystemState.SAFE


# -- no safe path -> PAUSED, "NO SAFE PATH", vehicle does not move --


def test_no_safe_path_forces_pause_with_explicit_message():
    engine = make_engine()
    result = engine.evaluate(
        perception_confidence=0.95,
        depth_confidence=0.95,
        localization_confidence=0.95,
        local_risk=0.0,
        goal_reachable=False,
        raw_min_clearance_m=None,
    )
    assert result.no_safe_path is True
    assert result.action == ActionCommand.PAUSE
    assert result.message == "NO SAFE PATH"


def test_pause_action_zeroes_cmd_vel_so_vehicle_does_not_move():
    engine = make_engine()
    linear, angular = engine.apply_action(ActionCommand.PAUSE, linear=1.5, angular=0.8)
    assert linear == 0.0
    assert angular == 0.0


# -- DEGRADED never silently resolves back to SAFE on its own --


def test_degraded_does_not_jump_straight_back_to_safe():
    engine = make_engine()
    # Drive into DEGRADED.
    degraded_result = engine.evaluate(
        perception_confidence=0.3,
        depth_confidence=0.3,
        localization_confidence=0.3,
        local_risk=0.0,
        goal_reachable=True,
        raw_min_clearance_m=None,
    )
    assert degraded_result.state == SystemState.DEGRADED

    # Conditions instantly look perfect again on the very next cycle.
    recovered_result = engine.evaluate(
        perception_confidence=0.99,
        depth_confidence=0.99,
        localization_confidence=0.99,
        local_risk=0.0,
        goal_reachable=True,
        raw_min_clearance_m=None,
    )
    assert recovered_result.state != SystemState.SAFE
    assert recovered_result.state == SystemState.WARNING  # only one severity level of recovery per cycle


def test_degraded_reaches_safe_only_after_passing_through_warning():
    engine = make_engine()
    engine.evaluate(0.3, 0.3, 0.3, 0.0, True, None)  # -> DEGRADED
    assert engine.current_state == SystemState.DEGRADED

    engine.evaluate(0.99, 0.99, 0.99, 0.0, True, None)  # -> WARNING (not SAFE)
    assert engine.current_state == SystemState.WARNING

    result = engine.evaluate(0.99, 0.99, 0.99, 0.0, True, None)  # -> SAFE, second good cycle
    assert result.state == SystemState.SAFE


def test_state_can_worsen_immediately_in_one_cycle():
    engine = make_engine()
    assert engine.current_state == SystemState.SAFE
    result = engine.evaluate(0.01, 0.01, 0.01, 0.0, True, None)
    assert result.state == SystemState.CRITICAL  # worsening is immediate, no staging


def test_reset_returns_to_safe():
    engine = make_engine()
    engine.evaluate(0.01, 0.01, 0.01, 0.0, True, None)
    assert engine.current_state == SystemState.CRITICAL
    engine.reset()
    assert engine.current_state == SystemState.SAFE


# -- raw-distance emergency backstop fires independent of risk-field classification --


def test_emergency_fires_even_when_risk_field_says_everything_is_fine():
    engine = make_engine()
    result = engine.evaluate(
        perception_confidence=0.99,  # vision stack fully confident
        depth_confidence=0.99,
        localization_confidence=0.99,
        local_risk=0.0,  # risk field never flagged this hazard (its close-range blind spot)
        goal_reachable=True,
        raw_min_clearance_m=0.1,  # but the raw proximity backstop reads something is right there
    )
    assert result.emergency_triggered is True
    assert result.state == SystemState.CRITICAL
    assert result.action == ActionCommand.PAUSE
    assert "EMERGENCY" in result.message


def test_emergency_check_does_not_consult_confidence_or_risk_at_all():
    engine = make_engine()
    assert engine.check_emergency(raw_min_clearance_m=0.05) is True
    assert engine.check_emergency(raw_min_clearance_m=5.0) is False
    assert engine.check_emergency(raw_min_clearance_m=None) is False


def test_emergency_overrides_even_a_currently_safe_state():
    engine = make_engine()
    engine.evaluate(0.99, 0.99, 0.99, 0.0, True, None)
    assert engine.current_state == SystemState.SAFE

    result = engine.evaluate(0.99, 0.99, 0.99, 0.0, True, raw_min_clearance_m=0.05)
    assert result.state == SystemState.CRITICAL
    assert result.action == ActionCommand.PAUSE


# -- state/action mapping sanity --


def test_state_action_mapping_is_the_deterministic_1to1_spec():
    engine = make_engine()
    assert engine.apply_action(ActionCommand.RUN, 1.0, 0.5) == (1.0, 0.5)
    slow_linear, slow_angular = engine.apply_action(ActionCommand.SLOW, 1.0, 0.5)
    assert 0 < slow_linear < 1.0
    assert slow_angular == 0.5
    assert engine.apply_action(ActionCommand.PAUSE, 1.0, 0.5) == (0.0, 0.0)
