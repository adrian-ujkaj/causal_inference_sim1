"""Fast simulation tests (wind, drag, formation, planner), without a full flight."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from environment.wind import DrydenGustModel  # noqa: E402
from Control.Path_planning import HeightmapAStar  # noqa: E402
from entities.uav import UAV  # noqa: E402
from swarm.swarm import Swarm  # noqa: E402


# Wind
def test_wind_has_dryden_intensity_and_correlation():
    """Gust intensity and correlation consistent with the Dryden model."""
    np.random.seed(3)
    dt, h, V = 0.01, 1.0, 5.0
    w = DrydenGustModel(dt, turbulence_intensity_knots=10)
    G = np.array([w.step(h, V, [V, 0.0, 0.0]) for _ in range(int(600 / dt))])
    sig, L = w._params(h, V)
    # intensity: W20 = 10 kt near the ground -> sigma_u ~ 1 m/s, sigma_w ~ 0.5 m/s
    assert 0.9 < sig[0] < 1.1 and 0.45 < sig[2] < 0.55
    assert np.allclose(G.std(axis=0), sig, rtol=0.25)
    # correlation at 1 s: exp(-V / L) for a Gauss-Markov process
    lag = int(1.0 / dt)
    for i in range(3):
        rho = np.corrcoef(G[:-lag, i], G[lag:, i])[0, 1]
        assert abs(rho - np.exp(-V * 1.0 / L[i])) < 0.1


def test_wind_burst_scales_same_realisation():
    """Burst over [t0, t1[: same draw, scaled intensity, then back to the base level."""
    dt, h, V = 0.01, 1.0, 2.0
    t = np.arange(0, 30, dt)
    base = DrydenGustModel(dt, turbulence_intensity_knots=10, seed=5)
    burst = DrydenGustModel(dt, turbulence_intensity_knots=10, seed=5, burst=(10.0, 20.0, 80.0))
    Gb = np.array([base.step(h, V, [V, 0, 0], t=tk) for tk in t])
    Gx = np.array([burst.step(h, V, [V, 0, 0], t=tk) for tk in t])
    before, during = t < 10, (t >= 10) & (t < 20)
    assert np.allclose(Gb[before], Gx[before])
    assert np.allclose(Gx[during], 8.0 * Gb[during])
    assert burst.turbulence_level == 10.0


def test_uav_wind_is_correlated_when_hovering():
    """Correlated gusts in hover, with the airspeed relative to the mean wind."""
    np.random.seed(2)
    dt = 1 / 240
    w = DrydenGustModel(dt, turbulence_intensity_knots=30)
    vel = np.zeros(3)
    G = np.array(
        [w.step(1.0, float(np.linalg.norm(vel - w.mean_wind)), vel - w.mean_wind) for _ in range(5000)]
    )
    rho = np.corrcoef(G[:-10, 0], G[10:, 0])[0, 1]
    assert rho > 0.9


def test_wind_is_reproducible_with_global_seed():
    np.random.seed(11)
    a = [DrydenGustModel(0.01, 10).step(1.0, 2.0) for _ in range(1)]
    np.random.seed(11)
    b = [DrydenGustModel(0.01, 10).step(1.0, 2.0) for _ in range(1)]
    assert np.allclose(a, b)


def test_wind_zero_intensity_returns_mean_wind():
    w = DrydenGustModel(0.01, 0.0, mean_wind=[1.0, -2.0, 0.0])
    assert np.allclose(w.step(5.0, 3.0), [1.0, -2.0, 0.0])


# Drag
@pytest.mark.parametrize("v0", [[-5.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, -4.0, 0.0], [1.0, 1.0, 0.0]])
@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 180.0, -135.0])
def test_drag_opposes_airspeed_for_any_yaw(yaw_deg, v0):
    """Drag opposes the airspeed, for several yaw angles and flight directions."""
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, 0, physicsClientId=cid)
        q = p.getQuaternionFromEuler([0, 0, np.radians(yaw_deg)])
        body = p.loadURDF(
            os.path.join(ROOT, "assets", "quadrotor.urdf"),
            [0, 0, 5],
            q,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=cid,
        )
        v0 = np.array(v0, dtype=float)
        p.resetBaseVelocity(body, v0.tolist(), [0, 0, 0], physicsClientId=cid)

        uav = UAV.__new__(UAV)  # minimal instance, without __init__
        uav.bodyId, uav.physics_client_id = body, cid
        uav.KF, uav.KM, uav.MAX_RPM = 0.0, 0.0, 22000.0  # no thrust: drag only
        uav.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])
        uav.current_wind = np.zeros(3)
        gt = dict(orn_q=q, vel=v0)
        uav._apply_lib_physics(np.full(4, 15000.0), gt)
        p.stepSimulation(physicsClientId=cid)
        v1, _ = p.getBaseVelocity(body, physicsClientId=cid)
        dv = np.array(v1) - v0
        assert np.dot(dv, v0) < 0.0
        cos = np.dot(dv, -v0) / (np.linalg.norm(dv) * np.linalg.norm(v0))
        assert cos > 0.95
    finally:
        p.disconnect(cid)


# Formation
def _fake(name, pos, yaw=0.0):
    return SimpleNamespace(name=name, start_pos=list(pos), start_orn=p.getQuaternionFromEuler([0, 0, yaw]))


def test_formation_followers_do_not_cross():
    sw = Swarm.__new__(Swarm)
    sw.leader = _fake("lead", [0, 0, 0])
    # order deliberately "crossed" in the list
    sw.followers = [_fake("left", [-1, 1, 0]), _fake("right", [-1, -1, 0])]
    sw.formation_body_offsets = {}
    sw._assign_default_triangular_offsets()
    off = sw.formation_body_offsets
    assert off["right"][1] < 0 < off["left"][1]  # each keeps its side
    assert np.all(np.array([off["left"][0], off["right"][0]]) < 0)
    sep = np.linalg.norm(off["left"] - off["right"])
    assert sep >= 1.0


def test_formation_respects_leader_heading():
    sw = Swarm.__new__(Swarm)
    sw.leader = _fake("lead", [0, 0, 0], yaw=np.pi / 2)
    # leader facing +y: world +y is forward, world -x is its left
    sw.followers = [_fake("a", [1, -1, 0]), _fake("b", [-1, -1, 0])]
    sw.formation_body_offsets = {}
    sw._assign_default_triangular_offsets()
    off = sw.formation_body_offsets
    assert off["b"][1] > 0 > off["a"][1]


# Planner
def test_astar_inflation_keeps_full_safety_margin():
    margin, res = 0.25, 0.25
    pl = HeightmapAStar(
        {"world_bounds": {"x": [-10, 10], "y": [-10, 10], "z": [0, 60]}, "safety_margin": margin},
        resolution=res,
    )
    # 6 m building centred at (0.1, 0.1): edges not aligned with the grid
    pl.build_from_buildings([{"center": [0.1, 0.1, 20.0], "height": 20.0, "width": 6.0, "length": 6.0}])
    free = pl.h_map < 1.0
    ii, jj = np.nonzero(free)
    xs0 = pl.min_x + ii * res
    ys0 = pl.min_y + jj * res
    # distance from each free cell (rectangle) to the real building
    dx = np.maximum(np.maximum(-2.9 - (xs0 + res), xs0 - 3.1), 0.0)
    dy = np.maximum(np.maximum(-2.9 - (ys0 + res), ys0 - 3.1), 0.0)
    clearance = np.hypot(dx, dy)
    assert clearance.min() >= margin - 1e-9


def test_astar_plans_from_ground_level():
    """A path is found from a start on the ground (z = 0)."""
    pl = HeightmapAStar(
        {"world_bounds": {"x": [-5, 5], "y": [-5, 5], "z": [0, 60]}, "safety_margin": 0.25}, resolution=0.25
    )
    pl.build_from_buildings([{"center": [0.0, 0.0, 10.0], "height": 10.0, "width": 2.0, "length": 2.0}])
    path = pl.plan(np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 1.0]))
    assert path is not None and len(path) > 2


# Reproducibility
def test_agent_seeds_follow_simulation_seed():
    """Same simulation.seed -> same sensor, wind and message streams for every drone."""
    from simulator.simulator_manager import _seed_agent

    def seeds(sim_seed):
        cfgs = [{"type": "uav", "sensors": {"imu": {}, "gnss": {}}, "wind": {}} for _ in range(3)]
        for cfg, stream in zip(cfgs, np.random.SeedSequence(sim_seed).spawn(3)):
            _seed_agent(cfg, stream)
        return [
            (c["sensors"]["imu"]["seed"], c["sensors"]["gnss"]["seed"], c["wind"]["seed"],
             c["gnss_schedule_seed"], c["comm_seed"])
            for c in cfgs
        ]

    assert seeds(7) == seeds(7)
    assert seeds(7) != seeds(8)
    flat = [s for drone in seeds(7) for s in drone]
    assert len(set(flat)) == len(flat)  # independent streams


def test_invalid_seed_fails_loudly():
    from simulator.simulator_manager import SimulationManager

    with pytest.raises(ValueError):
        SimulationManager({"simulation": {"dt": 0.00416, "seed": "12a", "connect_mode": "direct"}})
