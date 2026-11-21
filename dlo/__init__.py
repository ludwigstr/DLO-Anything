"""DLO-Anything (flat-surface version): MuJoCo cable simulation + physics transformer.

Importing this package before `mujoco` selects headless EGL rendering.
"""
import os as _os
_os.environ.setdefault("MUJOCO_GL", "egl")
