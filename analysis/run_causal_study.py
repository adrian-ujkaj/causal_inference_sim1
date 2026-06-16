"""Full causal study:
  1. three campaigns of paired flights (same seeds): base, gnss_fort (drone_1 GNSS
     jammed, noise x100 between 8 and 22 s) and vent_fort (turbulence 80 instead of 10 on
     drone_2, between 8 and 22 s);
  2. causal analysis of each campaign;
  3. validation: measured effect of each intervention, compared with the analysis conclusions.

The figures to look at are copied to runs/causal/RESULTATS/:
  1_verification.png        real effect / causes found by the analysis
  2_qui_influence_qui.png   NRI interaction graph (normal flights)
  3_causes_vol_normal.png   causes of failures in normal flight

    python analysis/run_causal_study.py                       # 24 flights x 30 s per campaign
    python analysis/run_causal_study.py --runs 8 --skip_sim   # analyses only

Allow 10 to 40 min per campaign depending on the machine."""

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CAMPAIGNS = {
    "base": None,
    "gnss_fort": {"@drone_1": {"sensors": {"gnss": {"jam_start": 8, "jam_end": 22, "jam_multiplier": 100}}}},
    "vent_fort": {"@drone_2": {"wind": {"burst_start": 8, "burst_end": 22, "burst_turbulence": 80}}},
}
WINDOWS = {"gnss_fort": "8:22", "vent_fort": "8:22"}


def run(cmd, log):
    print(">", " ".join(cmd))
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"echec ({r.returncode}), voir {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join("runs", "causal"))
    p.add_argument("--runs", type=int, default=24)
    p.add_argument("--tmax", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip_sim", action="store_true", help="reutiliser les vols deja simules")
    a = p.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    if not a.skip_sim:
        for name, ov in CAMPAIGNS.items():
            cmd = [
                PY,
                "analysis/filter_validation.py",
                "--runs",
                str(a.runs),
                "--tmax",
                str(a.tmax),
                "--workers",
                str(a.workers),
                "--logs",
                os.path.join(a.out, name),
                "--no-fig",
            ]
            if ov:
                cmd += ["--agent-override", json.dumps(ov)]
            run(cmd, os.path.join(out, f"{name}.log"))

    ana = os.path.join(a.out, "analyse")
    for name in CAMPAIGNS:
        run(
            [
                PY,
                "analysis/causal_analysis.py",
                "--log_dir",
                os.path.join(a.out, name),
                "--output_dir",
                os.path.join(ana, name),
                "--verbose_every",
                "0",
            ],
            os.path.join(out, f"analyse_{name}.log"),
        )

    cmd = [
        PY,
        "analysis/causal_validation.py",
        "--base",
        os.path.join(a.out, "base"),
        "--output_dir",
        os.path.join(a.out, "validation"),
    ]
    for name, w in WINDOWS.items():
        cmd += [
            "--intervention",
            f"{name}={os.path.join(a.out, name)}" + (f":{w}" if w else ""),
            "--analysis",
            f"{name}={os.path.join(ana, name)}",
        ]
    run(cmd, os.path.join(out, "validation.log"))
    print(open(os.path.join(out, "validation.log")).read())

    res = os.path.join(out, "RESULTATS")
    os.makedirs(res, exist_ok=True)
    for src, dst in (
        (os.path.join(out, "validation", "resume_etude.png"), "1_verification.png"),
        (os.path.join(ROOT, ana, "base", "graphe_interactions.png"), "2_qui_influence_qui.png"),
        (os.path.join(ROOT, ana, "base", "causes_defaillances.png"), "3_causes_vol_normal.png"),
    ):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(res, dst))
    print(f"Figures a regarder : {res}")


if __name__ == "__main__":
    main()
