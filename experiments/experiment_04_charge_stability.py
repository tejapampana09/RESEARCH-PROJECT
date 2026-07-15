"""
Experiment 04: Full Charge Stability Diagram Sweep
===================================================
Executes a high-resolution 2D plunger gate sweep generating the multi-panel
honeycomb charge stability diagram.

Saves:
  - QuantumTwin/results/exp04_charge_stability.png (.pdf)
  - QuantumTwin/results/exp04_charge_stability.csv
  - QuantumTwin/results/exp04_metadata.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from physics.charge_stability import ChargeStabilityDiagram

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 04: Full Charge Stability Diagram Sweep ---")
    np.random.seed(42)

    # 1. Physical parameters
    U_onsite = np.array([4.0e-3, 4.0e-3])     # 4.0 meV self-charging
    U_inter  = np.array([[0, 0.8e-3], [0.8e-3, 0]]) # 0.8 meV cross-charging
    lever_arms = np.array([[0.12, 0.02], [0.02, 0.12]]) # Cross-talk lever arm matrix

    csd = ChargeStabilityDiagram(U_onsite, U_inter, lever_arms, T=0.1, max_occupation=3)

    # 2. High-resolution grid sweep (50x50)
    Vg_x = np.linspace(0.00, 0.06, 50)
    Vg_y = np.linspace(0.00, 0.06, 50)
    result = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=True)

    # 3. Export CSV data
    csv_path = os.path.join(RESULTS_DIR, "exp04_charge_stability.csv")
    csd.export_csv(result, csv_path)

    # 4. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp04_metadata.json")
    meta = {
        "experiment": "04_charge_stability",
        "grid_resolution": [len(Vg_x), len(Vg_y)],
        "temperature_K": 0.1,
        "max_charge_state": int(result['charge_map'].max()),
        "triple_points_count": len(result['triple_points']),
        "triple_point_coords_mV": [(tp[0]*1e3, tp[1]*1e3) for tp in result['triple_points'][:5]]
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 5. Export publication figure using builtin plot method
    img_base = os.path.join(RESULTS_DIR, "exp04_charge_stability")
    csd.plot(result, title="Double Quantum Dot Honeycomb Stability Diagram",
             save_path=img_base, show_triple_points=True)
    print(f"Saved: {img_base}.png/.pdf")

if __name__ == '__main__':
    run()
