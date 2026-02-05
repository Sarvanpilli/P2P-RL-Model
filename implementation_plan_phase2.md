# Implementation Plan: Safety & Feasibility Layer (Phase 2)

## Goal
Implement a `FeasibilityFilter` to enforce hard physical constraints and prevent unsafe actions, regardless of the RL agent's output.

## Proposed Changes

### 1. [NEW] `train/safety_filter.py`

Create a new class `FeasibilityFilter` that takes `(action, state)` and returns `safe_action`.

**Constraints to Enforce:**
1.  **Simultaneous Charge/Discharge**: Ensure battery is not charging and discharging in the same step (physically impossible for single-port batteries).
2.  **SoC Limits**:
    *   If `SoC >= Max`, force `Charge = 0`.
    *   If `SoC <= Min`, force `Discharge = 0`.
3.  **Rate Limits**: Ensure power does not exceed `Max_kW`.

**Logic:**
*   Input: `raw_action` (from PPO).
*   Process:
    *   Clip to `[-Max, +Max]`.
    *   Check SoC. If full, clamp positive part to 0. If empty, clamp negative part to 0.
    *   (Optional) Grid limits check? (The env handles this via penalties, but a filter could enforce it strictly if required. For now, we'll stick to device safety).

### 2. [MODIFY] `train/energy_env_improved.py`

*   Import `FeasibilityFilter`.
*   In `step()`:
    *   Pass `action` through `FeasibilityFilter` *before* processing physics.
    *   Log `corrections` (difference between raw and safe action) in `info`.

## Verification Plan
*   Create a unit test `tests/test_safety.py` that feeds illegal actions (e.g., charge when full) and asserts the filter corrects them.
