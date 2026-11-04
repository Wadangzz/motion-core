"""Verification gates (design doc §9).

V3.5 lives here: ``mujoco_cross`` rebuilds the model as MJCF and compares the
RNE torques against MuJoCo's. MuJoCo is an optional dependency and is imported
inside the functions, never at module level, so importing ``motion_core`` never
requires it.
"""
