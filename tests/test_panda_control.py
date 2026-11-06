import numpy as np

from inspect_robots_libero.panda_control import compliant_target


def test_compliance_stays_inactive_until_force_in_target_direction():
    current = np.array([0.0, 0.0, 0.8])
    requested = current.copy()

    corrected, delta, active = compliant_target(current, requested, np.array([0, 0, 0.9]))

    np.testing.assert_array_equal(corrected, requested)
    assert delta == 0.0
    assert not active

    corrected, delta, active = compliant_target(current, requested, np.array([0, 0, -0.9]))

    np.testing.assert_array_equal(corrected, requested)
    assert delta == 0.0
    assert not active


def test_compliance_corrects_force_error_with_panda_specific_step_cap():
    current = np.array([0.0, 0.0, 0.8])
    requested = current.copy()

    over_target, delta, active = compliant_target(current, requested, np.array([0, 0, -4.0]))

    assert active
    assert np.isclose(delta, 0.0003)
    assert np.isclose(over_target[2], 0.8003)

    under_target, delta, active = compliant_target(current, requested, np.array([0, 0, -1.5]))

    assert active
    assert np.isclose(delta, -0.00015)
    assert np.isclose(under_target[2], 0.79985)


def test_compliance_correction_cannot_exceed_cartesian_step_or_workspace_limits():
    current = np.array([0.0, 0.0, 0.8])
    requested = np.array([0.0, 0.0, 0.805])

    corrected, _, _ = compliant_target(
        current, requested, np.array([0, 0, -5.0]), gain_m_per_n=1.0
    )

    assert corrected[2] <= current[2] + 0.01
    assert corrected[2] >= current[2] - 0.01
    assert 0.42 <= corrected[2] <= 1.6


def test_panda_compliant_axis_limits_each_requested_step_to_two_point_five_millimeters():
    current = np.array([0.0, 0.0, 0.8])
    requested = np.array([0.0, 0.0, 0.805])

    corrected, _, active = compliant_target(current, requested, np.zeros(3))

    assert not active
    assert np.isclose(corrected[2] - current[2], 0.0025)


def test_downward_contact_force_corrects_target_upward():
    current = np.array([0.0, 0.0, 0.8])
    requested = np.array([0.0, 0.0, 0.8])

    corrected, delta, active = compliant_target(
        current, requested, np.array([0.0, 0.0, -60.0])
    )

    assert active
    assert delta > 0.0
    assert np.isclose(corrected[2], current[2] + 0.0003)


def test_compliance_stays_latched_below_threshold_and_releases_on_reverse_motion():
    current = np.array([0.0, 0.0, 0.8])
    downward = np.array([0.0, 0.0, 0.8])

    held, _, active = compliant_target(
        current, downward, np.zeros(3), contact_latched=True
    )

    assert active
    assert held[2] < current[2]

    upward = np.array([0.0, 0.0, 0.81])
    released, _, active = compliant_target(
        current, upward, np.array([0.0, 0.0, -4.0]), contact_latched=True
    )

    assert not active
    assert released[2] > current[2]


def test_force_setpoint_is_zero_until_contact_then_ramps_and_latches_without_vision():
    from inspect_robots_libero.panda_control import force_setpoint_for_contact

    target, engaged, step = force_setpoint_for_contact(
        -2.0, np.zeros(3), ramp_step=0
    )
    assert target == 0.0 and not engaged and step == 0

    target, engaged, step = force_setpoint_for_contact(
        -2.0, np.array([0.0, 0.0, -1.2]), ramp_step=0
    )
    assert engaged and step == 1 and np.isclose(target, -0.1)

    target, engaged, step = force_setpoint_for_contact(
        -2.0, np.zeros(3), contact_latched=engaged, ramp_step=step
    )
    assert engaged and step == 2 and np.isclose(target, -0.2)

    target, engaged, step = force_setpoint_for_contact(
        -2.0, np.zeros(3), contact_latched=True, ramp_step=19
    )
    assert engaged and step == 20 and np.isclose(target, -2.0)
