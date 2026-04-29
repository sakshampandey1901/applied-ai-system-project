"""State machine and control-phrase extraction tests."""

from state_machine import (
    VoiceStateMachine,
    extract_control_command,
    State,
)


def test_extract_wake_sleep_shutdown():
    assert extract_control_command('Please say Atlas wake up now') == "wake"
    assert extract_control_command("Atlas go to sleep please") == "sleep"
    assert extract_control_command("Atlas shut down") == "shutdown"
    assert extract_control_command("Atlas close agent") == "shutdown"


def test_wake_transitions_idle_active():
    sm = VoiceStateMachine(initial=State.IDLE)
    assert sm.state == State.IDLE
    sm.wake()
    assert sm.state == State.ACTIVE


def test_sleep_transition_active_sleep():
    sm = VoiceStateMachine(initial=State.ACTIVE)
    sm.sleep()
    assert sm.state == State.SLEEP


def test_shutdown_always_reaches_terminal():
    sm = VoiceStateMachine(initial=State.IDLE)
    sm.shutdown()
    assert sm.state == State.SHUTDOWN


def test_sleep_from_explicit_command():
    sm = VoiceStateMachine(initial=State.ACTIVE)
    sm.apply_control_command("sleep")
    assert sm.state == State.SLEEP


def test_wake_from_sleep():
    sm = VoiceStateMachine(initial=State.SLEEP)
    sm.wake()
    assert sm.state == State.ACTIVE
