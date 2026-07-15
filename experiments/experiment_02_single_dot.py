"""
Experiment 02: Single Quantum Dot Characterization
===================================================
Simulates electrostatic potential profile, confinement potential wells, and
electronic energy levels for a single silicon quantum dot.

Saves:
  - QuantumTwin/results/exp02_single_dot.png (.pdf)
  - QuantumTwin/results/exp02_single_dot_data.csv
  - QuantumTwin/results/exp02_metadata.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from physics.potential import PotentialLandscape
from quantum.schrodinger import SchrodingerSolver2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 02: Single Quantum Dot ---")
    np.random.seed(42)

    # 1. Single dot landscape
    pot = PotentialLandscape(grid_size=(45, 45), extent=(80e-9, 80e-9))
    pot.add_dot("D1", x0=0.0, y0=0.0, depth=80e-3, radius=12e-9)
    pot.add_gate("P1", x0=25e-9, y0=0.0, lever_arm=0.12, sigma=15e-9)

    # Evaluate potential map for gate voltage P1 = 0.05 V (50 mV)
    V_map = pot.get_potential({"P1": 0.05})

    # 2. Solve 2D Schrödinger equation
    solver = SchrodingerSolver2D()
    evals_eV, psi_list = solver.solve(V_map, pot.x, pot.y, num_states=4)
    evals_meV = evals_eV * 1e3

    # 3. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp02_single_dot_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["State_Index", "Energy_meV", "Level_Spacing_meV"])
        for idx in range(len(evals_meV)):
            spacing = evals_meV[idx] - evals_meV[0] if idx > 0 else 0.0
            writer.writerow([idx, evals_meV[idx], spacing])
    print(f"Saved: {csv_path}")

    # 4. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp02_metadata.json")
    meta = {
        "experiment": "02_single_dot",
        "well_depth_meV": 80.0,
        "dot_radius_nm": 12.0,
        "plunger_gate_voltage_mV": 50.0,
        "ground_state_energy_meV": float(evals_meV[0]),
        "first_excited_energy_meV": float(evals_meV[1]),
        "level_spacing_meV": float(evals_meV[1] - evals_meV[0])
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 5. Plot figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    x_nm = pot.x * 1e9
    y_nm = pot.y * 1e9

    # Potential landscape contour
    im0 = axes[0].pcolormesh(x_nm, y_nm, V_map.T * 1e3, cmap='viridis_r', shading='auto')
    fig.colorbar(im0, ax=axes[0], label='Potential V(x,y) (meV)')
    axes[0].scatter([0], [0], color='red', marker='+', s=100, label='Dot center')
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('y (nm)')
    axes[0].set_title('Electrostatic Potential Well')
    axes[0].legend()

    # Potential cross-section along y=0 and energy levels
    mid_idx = pot.Nx // 2
    V_cut = V_map[:, mid_idx] * 1e3

    axes[1].plot(x_nm, V_cut, 'k-', lw=2, label='Potential V(x,0)')
    colors = ['royalblue', 'crimson', 'darkorange', 'seagreen']
    for idx, E in enumerate(evals_meV):
        axes[1].axhline(E, color=colors[idx % len(colors)], ls='--',
                        label=f'$E_{idx} = {E:.2f}$ meV')
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('Energy (meV)')
    axes[1].set_title('Cross-Section & Bound States')
    axes[1].legend(fontsize=9)

    img_base = os.path.join(RESULTS_DIR, "exp02_single_dot")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")
    print(f"Ground state = {evals_meV[0]:.2f} meV, 1st spacing = {evals_meV[1]-evals_meV[0]:.2f} meV")

if __name__ == '__main__':
    run()
