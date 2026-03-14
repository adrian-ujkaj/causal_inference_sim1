"""
Run a simulation described by a YAML configuration file.

    python main.py                  # config.yaml of the repository
    python main.py my_config.yaml   # another scenario
"""

import argparse
import os
import time

import pybullet as p

from simulator.simulator_manager import SimulationManager
from utilities.config import load_config

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(ROOT, "config.yaml"),
        help="fichier de configuration YAML (defaut : config.yaml du depot)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    os.chdir(ROOT)  # paths in the YAML (assets/..., logs/...) are relative to the repository
    config = load_config(config_path)
    sim = SimulationManager(config)
    try:
        sim.run()
        # In GUI mode, the window stays open at the end of the simulation.
        if str(config["simulation"]["connect_mode"]).lower() == "gui":
            print("Simulation terminee. Ferme la fenetre PyBullet pour quitter.")
            while p.isConnected():
                time.sleep(0.1)
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
