import numpy as np


class DrydenGustModel:
    """Low-altitude Dryden turbulence (MIL-F-8785C).

    Each component is a first-order Gauss-Markov process, with the standard deviation
    and correlation length of the model (W20 in knots, h in ft in the formulas).
    u follows the horizontal airspeed relative to the mean wind, w is vertical; output in
    the world frame. burst=(t0, t1, W20): stronger turbulence between t0 and t1 (same draw,
    scaled intensity)."""

    KNOT_TO_FTS = 1.68781
    FT_TO_M = 0.3048
    M_TO_FT = 3.28084

    def __init__(
        self,
        dt,
        turbulence_intensity_knots=15,
        mean_wind=(0.0, 0.0, 0.0),
        seed=None,
        min_airspeed=1.0,
        burst=None,
    ):
        self.dt = float(dt)
        self.turbulence_level = max(0.0, float(turbulence_intensity_knots))
        self.base_level = self.turbulence_level
        # (t0, t1, W20): intensity W20 between t0 and t1, None otherwise
        self.burst = None if burst is None else tuple(float(x) for x in burst)
        self.mean_wind = np.asarray(mean_wind, dtype=float).reshape(3)
        # V bounded from below (in hover, L / V would be infinite)
        self.min_airspeed = float(min_airspeed)

        # Dedicated generator, drawn from the global one unless a seed is given
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))

        self._gust = None  # [u, v, w] in m/s, wind frame
        self._heading = np.array([1.0, 0.0])
        self.last_sigmas = np.zeros(3)  # diagnostic

    def _params(self, h_m: float, V_ms: float):
        """Standard deviations [m/s] and correlation lengths [m] for (h, V)."""
        h = max(float(h_m) * self.M_TO_FT, 10.0)  # model validity: h >= 10 ft
        k = 0.177 + 0.000823 * min(h, 1000.0)
        sigma_w = 0.1 * self.turbulence_level * self.KNOT_TO_FTS  # ft/s
        sigma_u = sigma_w / k**0.4
        L_w = h
        L_u = h / k**1.2
        sig = np.array([sigma_u, sigma_u, sigma_w]) * self.FT_TO_M
        L = np.array([L_u, L_u, L_w]) * self.FT_TO_M
        return sig, L

    def _set_level(self, level: float) -> None:
        """Change the intensity; the current gust is rescaled."""
        level = max(0.0, float(level))
        if level == self.turbulence_level:
            return
        if self._gust is not None and self.turbulence_level > 0:
            self._gust = self._gust * (level / self.turbulence_level)
        else:
            self._gust = None  # reset at the next step in the steady-state regime
        self.turbulence_level = level

    def step(self, h_meters, V_ms, airspeed_vec=None, t=None):
        """Advance one step and return the total wind (mean + gust) in the world frame.
        airspeed_vec (ground velocity - mean wind) orients u; t is used for the burst window."""
        if self.burst is not None and t is not None:
            t0, t1, level = self.burst
            self._set_level(level if t0 <= t < t1 else self.base_level)
        V = max(float(V_ms), self.min_airspeed)
        sig, L = self._params(h_meters, V)
        self.last_sigmas = sig

        if airspeed_vec is not None:
            a = np.asarray(airspeed_vec, dtype=float).reshape(3)[:2]
            n = float(np.linalg.norm(a))
            if n > 0.2:
                self._heading = a / n

        if self._gust is None:
            # First step: draw in the steady-state regime
            self._gust = self._rng.normal(0.0, 1.0, 3) * sig
        else:
            a_k = np.exp(-V * self.dt / L)
            self._gust = a_k * self._gust + sig * np.sqrt(1.0 - a_k * a_k) * self._rng.normal(0.0, 1.0, 3)

        cx, cy = self._heading
        u, v, w = self._gust
        gust_world = np.array([cx * u - cy * v, cy * u + cx * v, w])
        return self.mean_wind + gust_world
