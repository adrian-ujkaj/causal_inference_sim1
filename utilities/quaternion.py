"""Quaternions for inertial navigation, with the PyBullet conventions:
[x, y, z, w] storage, Hamilton product, q = body -> world rotation, attitude
increment composed on the right: q(k+1) = q(k) (x) Exp(omega dt).
Sola (2017) stores [w, x, y, z]: same formulas, different order."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def skew(v) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x @ u = v x u."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    # Canonical representative (w >= 0): q and -q describe the same rotation.
    return q if q[3] >= 0.0 else -q


def mul(q1, q2) -> np.ndarray:
    """Hamilton product q1 (x) q2, [x, y, z, w] storage."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def conj(q) -> np.ndarray:
    """Conjugate = inverse for a unit quaternion."""
    x, y, z, w = np.asarray(q, dtype=float).reshape(4)
    return np.array([-x, -y, -z, w])


def exp(theta) -> np.ndarray:
    """
    Rotation vector (axis * angle, rad) -> unit quaternion.
    Taylor series near zero to avoid dividing by |theta|.
    """
    theta = np.asarray(theta, dtype=float).reshape(3)
    a = float(np.linalg.norm(theta))
    if a < 1e-8:
        # sin(a/2)/a ~ 1/2 - a^2/48
        q = np.array([*(0.5 * theta), 1.0 - a * a / 8.0])
        return q / np.linalg.norm(q)
    s = np.sin(0.5 * a) / a
    return np.array([*(s * theta), np.cos(0.5 * a)])


def log(q) -> np.ndarray:
    """Unit quaternion -> rotation vector (rad), angle in [0, pi]."""
    q = normalize(q)  # enforce w >= 0: shortest path
    v, w = q[:3], q[3]
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return 2.0 * v / max(w, _EPS)  # 2*v for small angles
    angle = 2.0 * np.arctan2(n, w)
    return angle * v / n


def to_rot(q) -> np.ndarray:
    """Quaternion [x, y, z, w] -> body -> world rotation matrix."""
    x, y, z, w = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_exp(theta) -> np.ndarray:
    """Rodrigues formula: rotation vector -> rotation matrix."""
    return to_rot(exp(theta))


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical interpolation between q0 and q1, t in [0, 1]."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = mul(conj(q0), q1)  # relative rotation q0 -> q1
    return normalize(mul(q0, exp(t * log(d))))


def from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (PyBullet convention, fixed rotations X then Y then Z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return normalize(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def to_euler(q) -> np.ndarray:
    """Quaternion -> [roll, pitch, yaw] (rad), same convention."""
    R = to_rot(q)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def attitude_error(q_est, q_true) -> np.ndarray:
    """Local attitude error dtheta such that q_true = q_est (x) Exp(dtheta) (ESKF convention)."""
    return log(mul(conj(q_est), q_true))
