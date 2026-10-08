#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_log_csv(path):
    """
    Load the CSV log written by the UAV and return a dict of np.array.
    Empty values are converted to NaN.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"Fichier vide ou sans header: {path}")

        data = {name: [] for name in fieldnames}

        for row in reader:
            for name in fieldnames:
                val = row.get(name, "")
                if val is None or val == "":
                    data[name].append(np.nan)
                else:
                    try:
                        data[name].append(float(val))
                    except ValueError:
                        data[name].append(np.nan)

    for k in data:
        data[k] = np.asarray(data[k], dtype=float)

    return data


def plot_log(path):
    data = load_log_csv(path)
    title = os.path.basename(path)

    # Column names of the uav.py log
    alias = {
        "t": "time",
        "x_true": "gt_x",
        "y_true": "gt_y",
        "z_true": "gt_z",
        "x_gps": "meas_x",
        "y_gps": "meas_y",
        "z_gps": "meas_z",
        "x_est": "est_x",
        "y_est": "est_y",
        "z_est": "est_z",
    }
    for old, new in alias.items():
        if old not in data and new in data:
            data[old] = data[new]

    t = data["t"]

    # Ground truth
    x_true = data["x_true"]
    y_true = data["y_true"]
    z_true = data["z_true"]

    # GPS (may be NaN if disabled)
    x_gps = data.get("x_gps", np.full_like(t, np.nan))
    y_gps = data.get("y_gps", np.full_like(t, np.nan))
    z_gps = data.get("z_gps", np.full_like(t, np.nan))

    # Filter estimate (may be NaN if disabled)
    x_est = data.get("x_est", np.full_like(t, np.nan))
    y_est = data.get("y_est", np.full_like(t, np.nan))
    z_est = data.get("z_est", np.full_like(t, np.nan))

    # ---------------- XY trajectory ----------------
    plt.figure(figsize=(6, 6))
    plt.plot(x_true, y_true, label="Vérité terrain", linewidth=2)

    if np.isfinite(x_gps).any():
        plt.scatter(x_gps, y_gps, s=4, alpha=0.5, label="GPS", marker="x")
    if np.isfinite(x_est).any():
        plt.plot(x_est, y_est, linestyle="--", label="Filtre")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectoire XY - {title}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # ---------------- Position error (norm) ----------------
    # GPS error
    err_gps = np.full_like(t, np.nan)
    if np.isfinite(x_gps).any():
        mask_gps = np.isfinite(x_gps) & np.isfinite(x_true)
        dx = x_gps - x_true
        dy = y_gps - y_true
        dz = z_gps - z_true
        err_norm = np.sqrt(dx**2 + dy**2 + dz**2)
        err_gps[mask_gps] = err_norm[mask_gps]

    # Filter error
    err_est = np.full_like(t, np.nan)
    if np.isfinite(x_est).any():
        mask_est = np.isfinite(x_est) & np.isfinite(x_true)
        dx = x_est - x_true
        dy = y_est - y_true
        dz = z_est - z_true
        err_norm = np.sqrt(dx**2 + dy**2 + dz**2)
        err_est[mask_est] = err_norm[mask_est]

    plt.figure(figsize=(8, 4))
    if np.isfinite(err_gps).any():
        plt.plot(t, err_gps, label="Erreur GPS", linewidth=1.5)
    if np.isfinite(err_est).any():
        plt.plot(t, err_est, label="Erreur filtre", linewidth=1.5)

    plt.xlabel("Temps [s]")
    plt.ylabel("||erreur position|| [m]")
    plt.title(f"Erreur de position - {title}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage : python analysis/plot_uav_log.py logs/drone_0.csv")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.isfile(log_path):
        print(f"Fichier introuvable : {log_path}")
        sys.exit(1)

    plot_log(log_path)


if __name__ == "__main__":
    main()
