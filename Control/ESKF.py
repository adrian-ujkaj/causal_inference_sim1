"""15-state ESKF for inertial / GNSS integration (Sola 2017, local angular error).
Error dx = [dp, dv, dtheta, db_a, db_g]; p, v in the world frame, biases in the body frame.
Body -> world q stored as [x, y, z, w] (Sola: [w, x, y, z]), q_true = q (x) Exp(dtheta).
"""

from __future__ import annotations

import numpy as np

from utilities import quaternion as Q

_P = slice(0, 3)
_V = slice(3, 6)
_TH = slice(6, 9)
_BA = slice(9, 12)
_BG = slice(12, 15)


class ESKF:
    STATE_DIM = 15
    name = "eskf"

    def __init__(
        self, dt, gnss_config: dict | None = None, imu_config: dict | None = None, config: dict | None = None
    ):
        cfg = dict(config or {})
        imu = dict(imu_config or {})
        gnss = dict(gnss_config or {})

        self.dt = float(dt)
        self.g = float(cfg.get("gravity", imu.get("gravity", 9.81)))
        self.g_vec = np.array([0.0, 0.0, -self.g])

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.aligned = False
        # Initial heading assumed known (compass), with uncertainty init_yaw_std_deg
        self.initial_yaw = float(cfg.get("initial_yaw", 0.0))

        self._a_std = float(imu.get("accel_noise_std", 0.0))
        self._a_den = imu.get("accel_noise_density", None)
        self._g_std = float(imu.get("gyro_noise_std", 0.0))
        self._g_den = imu.get("gyro_noise_density", None)
        # Floors: zero noise would make P singular and the filter over-confident
        self._a_floor = float(cfg.get("accel_noise_floor", 0.02))
        self._g_floor = float(cfg.get("gyro_noise_floor", 0.002))
        # Margin for what the model ignores (scale factors, integration)
        self._a_margin = float(cfg.get("accel_model_margin", 0.05))
        self._g_margin = float(cfg.get("gyro_model_margin", 0.0))

        # Non-zero floor, otherwise P_bias goes to 0 and biases are no longer corrected
        self.sig_ba_rw = max(
            float(imu.get("accel_bias_rw", 0.0)), float(cfg.get("accel_bias_rw_floor", 1e-3))
        )
        self.sig_bg_rw = max(float(imu.get("gyro_bias_rw", 0.0)), float(cfg.get("gyro_bias_rw_floor", 1e-4)))

        sp0 = float(cfg.get("init_pos_std", 0.5))
        sv0 = float(cfg.get("init_vel_std", 0.1))
        srp0 = np.radians(float(cfg.get("init_roll_pitch_std_deg", 3.0)))
        syaw0 = np.radians(float(cfg.get("init_yaw_std_deg", 5.0)))
        # Plausible bias: turn-on bias and constant offset declared by the sensor
        sba0 = float(
            cfg.get(
                "init_accel_bias_std",
                np.sqrt(
                    float(imu.get("accel_bias_std", 0.0)) ** 2
                    + float(imu.get("accel_noise_mean", 0.0)) ** 2
                    + 0.05**2
                ),
            )
        )
        sbg0 = float(cfg.get("init_gyro_bias_std", np.hypot(float(imu.get("gyro_bias_std", 0.0)), 0.005)))
        self.P = np.diag(
            [sp0**2] * 3 + [sv0**2] * 3 + [srp0**2, srp0**2, syaw0**2] + [sba0**2] * 3 + [sbg0**2] * 3
        )

        self.sigma_p = float(gnss.get("position_noise_std", 0.1)) or 0.1
        self.sigma_v = float(gnss.get("velocity_noise_std", 0.05)) or 0.05
        self.R = np.diag([self.sigma_p**2] * 3 + [self.sigma_v**2] * 3)

        self.nis_gate = cfg.get("nis_gate", 22.46)  # chi2(6), p = 0.001
        self.max_consecutive_rejected = int(cfg.get("max_consecutive_rejected", 5))
        self.n_consecutive_rejected = 0

        self.last_innovation = np.zeros(6)
        self.last_S = np.zeros((6, 6))
        self.last_nis = np.nan
        self.n_updates = 0
        self.n_rejected = 0
        self.n_gate_recoveries = 0
        self.last_acc_world = np.zeros(3)

    # Common interface with INSGNSSFilter
    @property
    def position(self) -> np.ndarray:
        return self.p

    @property
    def velocity(self) -> np.ndarray:
        return self.v

    @property
    def attitude(self) -> np.ndarray:
        return self.q

    @property
    def x(self) -> np.ndarray:
        """Read-only [p, v], like INSGNSSFilter.x."""
        return np.concatenate([self.p, self.v])

    def init_state(self, position, velocity=None, attitude=None):
        self.p = np.asarray(position, dtype=float).reshape(3).copy()
        self.v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3).copy()
        if attitude is not None:
            self.q = Q.normalize(attitude)

    def sigmas(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    def align(self, acc_body, yaw: float = 0.0) -> None:
        """Coarse alignment at rest: roll and pitch from gravity, yaw given."""
        f = np.asarray(acc_body, dtype=float).reshape(3)
        roll = np.arctan2(f[1], f[2])
        pitch = np.arctan2(-f[0], np.hypot(f[1], f[2]))
        self.q = Q.from_euler(roll, pitch, yaw)
        self.aligned = True

    def _noise_std(self, std, density, floor, margin, dt) -> float:
        if density is not None and float(density) > 0.0 and dt > 0.0:
            s = float(density) / np.sqrt(dt)
        else:
            s = float(std)
        return float(np.sqrt(max(s, floor) ** 2 + margin**2))

    def predict(self, acc_body, gyro_body, dt: float | None = None) -> None:
        """Propagation with the inertial measurements over the interval dt."""
        dt = self.dt if (dt is None or float(dt) <= 0.0) else float(dt)
        a_m = np.asarray(acc_body, dtype=float).reshape(3)
        w_m = np.asarray(gyro_body, dtype=float).reshape(3)

        if not self.aligned:
            # First sample: drone on the ground, the accelerometer only measures gravity
            self.align(a_m, yaw=self.initial_yaw)

        w = w_m - self.bg
        a_b = a_m - self.ba

        # Nominal state
        q_mid = Q.mul(self.q, Q.exp(0.5 * w * dt))
        R_mid = Q.to_rot(q_mid)
        a_w = R_mid @ a_b + self.g_vec
        self.last_acc_world = a_w

        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = Q.normalize(Q.mul(self.q, Q.exp(w * dt)))

        # Error-state transition
        I3 = np.eye(3)
        RA = R_mid @ Q.skew(a_b)
        F = np.eye(15)
        F[_P, _V] = I3 * dt
        F[_P, _TH] = -0.5 * RA * dt * dt
        F[_P, _BA] = -0.5 * R_mid * dt * dt
        F[_V, _TH] = -RA * dt
        F[_V, _BA] = -R_mid * dt
        F[_TH, _TH] = Q.rot_exp(w * dt).T
        F[_TH, _BG] = -I3 * dt

        sa = self._noise_std(self._a_std, self._a_den, self._a_floor, self._a_margin, dt)
        sg = self._noise_std(self._g_std, self._g_den, self._g_floor, self._g_margin, dt)
        qa = sa * sa
        Qd = np.zeros((15, 15))
        # accelerometer noise: acts on v (dt) and on p (dt^2/2), correlated
        Qd[_P, _P] = qa * (dt**4) / 4.0 * I3
        Qd[_P, _V] = qa * (dt**3) / 2.0 * I3
        Qd[_V, _P] = qa * (dt**3) / 2.0 * I3
        Qd[_V, _V] = qa * (dt**2) * I3
        Qd[_TH, _TH] = sg * sg * (dt**2) * I3
        Qd[_BA, _BA] = self.sig_ba_rw**2 * dt * I3
        Qd[_BG, _BG] = self.sig_bg_rw**2 * dt * I3

        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, pos_meas, vel_meas=None, pos_std: float | None = None, vel_std: float | None = None):
        """GNSS correction (vel_meas optional), pos_std / vel_std given by the receiver. Returns (p, v)."""
        use_vel = vel_meas is not None
        m = 6 if use_vel else 3
        sp = self.sigma_p if pos_std is None else max(float(pos_std), 1e-6)
        sv = self.sigma_v if vel_std is None else max(float(vel_std), 1e-6)

        H = np.zeros((m, 15))
        H[0:3, _P] = np.eye(3)
        if use_vel:
            H[3:6, _V] = np.eye(3)
            z = np.hstack([np.asarray(pos_meas, float), np.asarray(vel_meas, float)])
            h = np.hstack([self.p, self.v])
            R = np.diag([sp * sp] * 3 + [sv * sv] * 3)
        else:
            z = np.asarray(pos_meas, float).reshape(3)
            h = self.p.copy()
            R = np.diag([sp * sp] * 3)

        y = z - h
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            nis = float(y @ np.linalg.solve(S, y))
            K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1
        except np.linalg.LinAlgError:
            print("[ESKF] S singuliere : mise a jour ignoree (symptome de divergence)")
            return self.p.copy(), self.v.copy()

        self.last_innovation = y
        self.last_S = S
        self.last_nis = nis
        self.n_updates += 1

        gate = self.nis_gate
        if gate and m != 6:
            gate = float(gate) * m / 6.0  # coarse threshold adjustment
        if gate and nis > float(gate):
            self.n_consecutive_rejected += 1
            if self.n_consecutive_rejected <= self.max_consecutive_rejected:
                self.n_rejected += 1
                return self.p.copy(), self.v.copy()
            # Too many rejections in a row: the filter is wrong, not the measurement
            self.n_gate_recoveries += 1
            self.P = self.P * 10.0
            S = H @ self.P @ H.T + R
            S = 0.5 * (S + S.T)
            K = np.linalg.solve(S, H @ self.P).T
        self.n_consecutive_rejected = 0

        # Error correction, Joseph-form covariance
        dx = K @ y
        I15 = np.eye(15)
        A = I15 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

        # Injection into the nominal state
        self.p = self.p + dx[_P]
        self.v = self.v + dx[_V]
        self.q = Q.normalize(Q.mul(self.q, Q.exp(dx[_TH])))
        self.ba = self.ba + dx[_BA]
        self.bg = self.bg + dx[_BG]

        # Error reset (reset Jacobian on the attitude)
        G = np.eye(15)
        G[_TH, _TH] = np.eye(3) - Q.skew(0.5 * dx[_TH])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.p.copy(), self.v.copy()

    def error_vector(self, p_true, v_true, q_true=None, ba_true=None, bg_true=None):
        """'True minus estimated' error in the error-state convention."""
        e = np.full(15, np.nan)
        e[_P] = np.asarray(p_true, float) - self.p
        e[_V] = np.asarray(v_true, float) - self.v
        if q_true is not None:
            e[_TH] = Q.attitude_error(self.q, q_true)
        if ba_true is not None:
            e[_BA] = np.asarray(ba_true, float) - self.ba
        if bg_true is not None:
            e[_BG] = np.asarray(bg_true, float) - self.bg
        return e

    def nees(self, p_true, v_true, q_true=None, ba_true=None, bg_true=None):
        """Returns (position-velocity NEES in dim 6, full NEES in dim 15 or NaN if some truth is missing)."""
        e = self.error_vector(p_true, v_true, q_true, ba_true, bg_true)
        try:
            e6 = e[0:6]
            nees_pv = float(e6 @ np.linalg.solve(self.P[0:6, 0:6], e6))
        except np.linalg.LinAlgError:
            nees_pv = float("nan")
        if np.any(np.isnan(e)):
            return nees_pv, float("nan")
        try:
            nees_full = float(e @ np.linalg.solve(self.P, e))
        except np.linalg.LinAlgError:
            nees_full = float("nan")
        return nees_pv, nees_full
