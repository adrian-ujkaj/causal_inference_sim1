import numpy as np

from utilities import quaternion as Q


class Sensor:
    """Base class for sensors."""

    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée.")


class GNSSensor(Sensor):
    """Simulated GNSS: noisy position [m] and velocity [m/s], jamming (jam_*) and outages (outages).
    last_pos_std / last_vel_std give the applied standard deviation, like the hAcc / sAcc of a u-blox receiver.
    """

    def __init__(self, config: dict):
        super().__init__()
        config = dict(config or {})

        self.pos_noise_std = max(0.0, float(config.get("position_noise_std", 0.0)))
        self.vel_noise_std = max(0.0, float(config.get("velocity_noise_std", 0.0)))
        self.pos_noise_std_base = self.pos_noise_std
        self.vel_noise_std_base = self.vel_noise_std

        # Jamming
        self.jam_start = config.get("jam_start", None)
        self.jam_end = config.get("jam_end", None)
        self.jam_pos_noise_std = config.get("jam_position_noise_std", config.get("jam_pos_noise_std", None))
        self.jam_vel_noise_std = config.get("jam_velocity_noise_std", config.get("jam_vel_noise_std", None))
        self.jam_multiplier = float(config.get("jam_multiplier", 1.0))

        # Outages
        windows = list(config.get("outages", []) or [])
        if config.get("outage_start") is not None and config.get("outage_end") is not None:
            windows.append([config["outage_start"], config["outage_end"]])
        self.outages = [(float(a), float(b)) for a, b in windows]

        seed = config.get("seed", None)
        self._rng = None if seed is None else np.random.default_rng(int(seed))

        self.last_pos_std = self.pos_noise_std
        self.last_vel_std = self.vel_noise_std
        self.available = True

    def is_available(self, t: float | None) -> bool:
        """False during an outage window."""
        if t is None:
            return True
        return not any(a <= float(t) <= b for a, b in self.outages)

    def _normal(self, std: float) -> np.ndarray:
        if self._rng is not None:
            return self._rng.normal(0.0, std, 3)
        return np.random.normal(0.0, std, 3)

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray,
        t: float | None = None,
    ):
        """Returns noisy (position, velocity), or (None, None) during an outage."""
        self.available = self.is_available(t)
        if not self.available:
            return None, None

        pos_std = self.pos_noise_std
        vel_std = self.vel_noise_std

        if (t is not None) and (self.jam_start is not None) and (self.jam_end is not None):
            if float(self.jam_start) <= float(t) <= float(self.jam_end):
                if self.jam_pos_noise_std is not None:
                    pos_std = float(self.jam_pos_noise_std)
                else:
                    pos_std = float(self.pos_noise_std_base) * float(self.jam_multiplier)
                if self.jam_vel_noise_std is not None:
                    vel_std = float(self.jam_vel_noise_std)
                else:
                    vel_std = float(self.vel_noise_std_base) * float(self.jam_multiplier)

        self.last_pos_std = pos_std
        self.last_vel_std = vel_std

        meas_pos = np.asarray(ground_truth_position, dtype=float) + self._normal(pos_std)
        meas_vel = np.asarray(ground_truth_velocity, dtype=float) + self._normal(vel_std)
        return meas_pos, meas_vel


class IMUSensor(Sensor):
    """Simulated MEMS IMU, in the body frame: measurement = (1 + s) * true + bias (constant + random walk) + noise.
    Values averaged over [k-1, k]: gyro = Log(q(k-1)^-1 (x) q(k)) / dt, accel projected at mid-interval.
    accel_error_det, gyro_error_det and q_mid give the ground truth of the errors for the NEES.
    """

    def __init__(self, config: dict | None = None, dt: float | None = None):
        super().__init__()
        cfg = dict(config or {})

        dt_cfg = cfg.get("dt", dt)
        if dt_cfg is None or float(dt_cfg) <= 0.0:
            dt_cfg = 1.0 / 100.0
        self.dt_nominal = float(dt_cfg)

        g = float(cfg.get("gravity", 9.81))
        self.g_vector = np.array([0.0, 0.0, g], dtype=float)

        # Dedicated generator, reproducible draws in Monte-Carlo
        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))

        # Accelerometer: noise [m/s^2], density [m/s^2/sqrt(Hz)], random walk [m/s^2/sqrt(s)]
        self.accel_noise_std = max(0.0, float(cfg.get("accel_noise_std", 0.0)))
        self.accel_noise_density = cfg.get("accel_noise_density", None)
        self.accel_noise_mean = float(cfg.get("accel_noise_mean", 0.0))
        self.accel_bias_rw = max(0.0, float(cfg.get("accel_bias_rw", 0.0)))
        accel_bias_std = max(0.0, float(cfg.get("accel_bias_std", 0.0)))
        accel_scale_std = max(0.0, float(cfg.get("accel_scale_std", 0.0)))

        # Gyroscope: same quantities in rad/s
        self.gyro_noise_std = max(0.0, float(cfg.get("gyro_noise_std", 0.0)))
        self.gyro_noise_density = cfg.get("gyro_noise_density", None)
        self.gyro_bias_rw = max(0.0, float(cfg.get("gyro_bias_rw", 0.0)))
        gyro_bias_std = max(0.0, float(cfg.get("gyro_bias_std", 0.0)))
        gyro_scale_std = max(0.0, float(cfg.get("gyro_scale_std", 0.0)))

        # Drawn once per sensor (turn-on bias)
        self.accel_bias = self._rng.normal(0.0, accel_bias_std, 3) if accel_bias_std > 0 else np.zeros(3)
        self.gyro_bias = self._rng.normal(0.0, gyro_bias_std, 3) if gyro_bias_std > 0 else np.zeros(3)
        self.accel_scale = self._rng.normal(0.0, accel_scale_std, 3) if accel_scale_std > 0 else np.zeros(3)
        self.gyro_scale = self._rng.normal(0.0, gyro_scale_std, 3) if gyro_scale_std > 0 else np.zeros(3)

        self.accel_bias_0 = self.accel_bias.copy()
        self.gyro_bias_0 = self.gyro_bias.copy()

        self.last_vel = None  # None on the first call: acceleration assumed zero
        self.last_q = None

        # Ground truth of the errors
        self.accel_error_det = self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_bias.copy()
        self.q_mid = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_acc_body_true = np.zeros(3)
        self.last_gyro_body_true = np.zeros(3)

    def reset(self, vel: np.ndarray | None = None, orn_q=None) -> None:
        """Reset the differentiation (between two Monte-Carlo runs)."""
        self.last_vel = None if vel is None else np.asarray(vel, dtype=float).copy()
        self.last_q = None if orn_q is None else Q.normalize(orn_q)

    def _white_std(self, std: float, density, dt: float) -> float:
        if density is not None:
            d = float(density)
            if d > 0.0 and dt > 0.0:
                return d / np.sqrt(dt)
            return 0.0
        return std

    def measure(self, vel, orn_q, ang_vel=None, dt: float | None = None):
        """Returns (acc_body, gyro_body); vel in the world frame [m/s], orn_q body -> world, ang_vel ignored."""
        step = self.dt_nominal if (dt is None or float(dt) <= 0.0) else float(dt)

        vel = np.asarray(vel, dtype=float).reshape(3)
        q_k = Q.normalize(orn_q)
        q_prev = q_k if self.last_q is None else self.last_q

        if self.last_vel is None:
            acc_world = np.zeros(3)
        else:
            acc_world = (vel - self.last_vel) / step
        self.last_vel = vel.copy()
        self.last_q = q_k

        # Specific force f = a - g_vec, with g_vec = [0, 0, -g]
        acc_proper_world = acc_world + self.g_vector

        # World -> body projection at mid-interval, like the ESKF mechanisation
        self.q_mid = Q.slerp(q_prev, q_k, 0.5)
        acc_body_true = Q.to_rot(self.q_mid).T @ acc_proper_world

        # Mean angular velocity that exactly reproduces the rotation over the interval
        gyro_body_true = Q.log(Q.mul(Q.conj(q_prev), q_k)) / step

        self.last_acc_body_true = acc_body_true
        self.last_gyro_body_true = gyro_body_true

        if self.accel_bias_rw > 0.0:
            self.accel_bias = self.accel_bias + self._rng.normal(0.0, self.accel_bias_rw * np.sqrt(step), 3)
        if self.gyro_bias_rw > 0.0:
            self.gyro_bias = self.gyro_bias + self._rng.normal(0.0, self.gyro_bias_rw * np.sqrt(step), 3)

        self.accel_error_det = self.accel_scale * acc_body_true + self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_scale * gyro_body_true + self.gyro_bias

        a_std = self._white_std(self.accel_noise_std, self.accel_noise_density, step)
        acc_body = acc_body_true + self.accel_error_det
        if a_std > 0.0:
            acc_body = acc_body + self._rng.normal(0.0, a_std, 3)

        g_std = self._white_std(self.gyro_noise_std, self.gyro_noise_density, step)
        gyro_body = gyro_body_true + self.gyro_error_det
        if g_std > 0.0:
            gyro_body = gyro_body + self._rng.normal(0.0, g_std, 3)

        return acc_body, gyro_body
