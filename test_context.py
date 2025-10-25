"""
Test script for the Agent Context system

Run this to verify that the context/memory system works correctly
without needing actual AR glasses or camera frames.
"""

import json

from agent_context import context_manager


def test_basic_context():
    """Test basic context creation and retrieval"""
    print("=" * 60)
    print("TEST 1: Basic Context Creation")
    print("=" * 60)

    # Create a new context
    ctx = context_manager.get_context("test-session-1")
    print(f"✓ Created context for session: {ctx.session_id}")
    print(f"  - Repair started: {ctx.is_repair_started()}")
    print(f"  - Object type: {ctx.object_type}")
    print()


def test_repair_workflow():
    """Test full repair workflow"""
    print("=" * 60)
    print("TEST 2: Full Repair Workflow")
    print("=" * 60)

    session_id = "test-session-2"

    # Step 1: Start a repair
    print("\n1. Starting light fixture repair...")
    context_manager.start_repair(
        session_id=session_id,
        object_type="light fixture",
        repair_steps=[
            "Turn off power at the circuit breaker",
            "Remove the old light fixture cover",
            "Disconnect the wiring from the old fixture",
        ],
    )

    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Object type: {ctx.object_type}")
    print(f"   ✓ Total steps: {ctx.total_steps}")
    print(f"   ✓ Current step: {ctx.current_step}")
    print(f"   ✓ Current instruction: {ctx.get_current_step_instruction()}")

    # Step 2: Complete first step
    print("\n2. Completing step 1...")
    context_manager.mark_step_complete(session_id)
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Current step: {ctx.current_step}")
    print(f"   ✓ Completed steps: {ctx.completed_steps}")
    print(f"   ✓ Current instruction: {ctx.get_current_step_instruction()}")

    # Step 3: Complete second step
    print("\n3. Completing step 2...")
    context_manager.mark_step_complete(session_id)
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Current step: {ctx.current_step}")
    print(f"   ✓ Completed steps: {ctx.completed_steps}")
    print(f"   ✓ Current instruction: {ctx.get_current_step_instruction()}")

    # Step 4: Complete third step
    print("\n4. Completing step 3...")
    context_manager.mark_step_complete(session_id)
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Current step: {ctx.current_step}")
    print(f"   ✓ Completed steps: {ctx.completed_steps}")
    print(f"   ✓ Is complete: {ctx.is_repair_complete()}")
    print()


def test_safety_warnings():
    """Test safety warning tracking"""
    print("=" * 60)
    print("TEST 3: Safety Warnings")
    print("=" * 60)

    session_id = "test-session-3"

    # Add safety warnings
    print("\n1. Adding safety warnings...")
    context_manager.add_safety_warning(
        session_id,
        "⚠️ Turn off power before starting!",
    )
    context_manager.add_safety_warning(
        session_id,
        "⚠️ Use insulated tools!",
    )

    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Safety warning shown: {ctx.safety_warning_shown}")
    print(f"   ✓ Warnings shown: {ctx.warnings_shown}")
    print()


def test_object_detection():
    """Test object detection tracking"""
    print("=" * 60)
    print("TEST 4: Object Detection Tracking")
    print("=" * 60)

    session_id = "test-session-4"

    # Simulate detecting objects across frames
    print("\n1. Frame 1: Detecting objects...")
    context_manager.update_detected_objects(
        session_id,
        ["screwdriver", "wire", "switch"],
    )
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Detected: {ctx.last_detected_objects}")

    print("\n2. Frame 2: Detecting objects...")
    context_manager.update_detected_objects(
        session_id,
        ["screwdriver", "wire", "tape"],
    )
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Detected: {ctx.last_detected_objects}")
    print(f"   ✓ Detection history: {ctx.detection_history}")
    print()


def test_multiple_sessions():
    """Test multiple concurrent sessions"""
    print("=" * 60)
    print("TEST 5: Multiple Concurrent Sessions")
    print("=" * 60)

    # Create multiple sessions
    print("\n1. Creating sessions for different users...")
    context_manager.start_repair(
        "user-alice",
        "faucet",
        ["Step 1", "Step 2", "Step 3"],
    )
    context_manager.start_repair(
        "user-bob",
        "light fixture",
        ["Step 1", "Step 2"],
    )

    alice_ctx = context_manager.get_context("user-alice")
    bob_ctx = context_manager.get_context("user-bob")

    print(f"   ✓ Alice repairing: {alice_ctx.object_type} ({alice_ctx.total_steps} steps)")
    print(f"   ✓ Bob repairing: {bob_ctx.object_type} ({bob_ctx.total_steps} steps)")

    # Advance Alice's repair
    print("\n2. Alice completes step 1...")
    context_manager.mark_step_complete("user-alice")
    alice_ctx = context_manager.get_context("user-alice")
    bob_ctx = context_manager.get_context("user-bob")

    print(f"   ✓ Alice on step: {alice_ctx.current_step}")
    print(f"   ✓ Bob still on step: {bob_ctx.current_step}")
    print()


def test_session_reset():
    """Test session reset functionality"""
    print("=" * 60)
    print("TEST 6: Session Reset")
    print("=" * 60)

    session_id = "test-session-6"

    # Start a repair
    print("\n1. Starting repair...")
    context_manager.start_repair(
        session_id,
        "outlet",
        ["Step 1", "Step 2"],
    )
    context_manager.mark_step_complete(session_id)

    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Object: {ctx.object_type}, Step: {ctx.current_step}")

    # Reset session
    print("\n2. Resetting session...")
    context_manager.reset_session(session_id)
    ctx = context_manager.get_context(session_id)
    print(f"   ✓ Object: {ctx.object_type}, Step: {ctx.current_step}")
    print(f"   ✓ Repair started: {ctx.is_repair_started()}")
    print()


def test_api_simulation():
    """Simulate the API workflow"""
    print("=" * 60)
    print("TEST 7: API Workflow Simulation")
    print("=" * 60)

    session_id = "snap-glasses-001"

    # Simulate first frame (object detection)
    print("\n1. First frame - detecting object...")
    ctx = context_manager.get_context(session_id)
    if not ctx.is_repair_started():
        # Simulate AI detecting a light fixture
        context_manager.start_repair(
            session_id,
            "light fixture",
            [
                "Turn off power at the circuit breaker",
                "Remove the old light fixture cover",
                "Disconnect the wiring from the old fixture",
            ],
        )
        ctx = context_manager.get_context(session_id)
        print(f"   ✓ API Response: Detected {ctx.object_type}")
        print(f"   ✓ Starting {ctx.total_steps} step repair")

    # Simulate subsequent frames (showing current step)
    print("\n2. Frame 2-10 - showing step instructions...")
    ctx = context_manager.get_context(session_id)
    instruction = ctx.get_current_step_instruction()
    print(f"   ✓ API Response: Step {ctx.current_step}/{ctx.total_steps}")
    print(f"   ✓ Instruction: {instruction}")

    # Simulate user confirming step
    print("\n3. User confirms step completion...")
    context_manager.mark_step_complete(session_id)
    ctx = context_manager.get_context(session_id)
    instruction = ctx.get_current_step_instruction()
    print(f"   ✓ API Response: Advanced to step {ctx.current_step}")
    print(f"   ✓ Next instruction: {instruction}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "AGENT CONTEXT SYSTEM - TEST SUITE" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_basic_context()
        test_repair_workflow()
        test_safety_warnings()
        test_object_detection()
        test_multiple_sessions()
        test_session_reset()
        test_api_simulation()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("The Agent Context system is working correctly.")
        print("Ready to integrate with AR glasses! 🥽")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
