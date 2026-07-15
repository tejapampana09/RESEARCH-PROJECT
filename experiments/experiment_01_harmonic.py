"""
Experiment 01: 2D Harmonic Oscillator Verification
===================================================
Validates the 2D finite-difference Schrödinger solver against analytical
harmonic oscillator energy levels E_(nx,ny) = ℏω (nx + ny + 1).

Saves:
  - QuantumTwin/results/exp01_wavefunction.png (.pdf)
  - QuantumTwin/results/exp01_energy_levels.csv
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from quantum.schrodinger import SchrodingerSolver2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 01: 2D Harmonic Oscillator ---")
    np.random.seed(42)

    # 1. Harmonic potential parameters
    m_eff = 0.19 * 9.1093837015e-31  # Si transverse effective mass
    omega = 1.5e12                  # Confinement frequency (rad/s)
    hbar  = 1.054571817e-34          # J s
    e_c   = 1.602176634e-19          # C

    # Theoretical ℏω in meV
    hbar_omega_meV = (hbar * omega / e_c) * 1e3   # ~0.987 meV

    # Grid setup: 45x45 grid over [-40nm, 40nm]
    L = 80e-9
    N = 45
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # V(x,y) = 0.5 * m* * w^2 * (x^2 + y^2) in eV
    V_eV = (0.5 * m_eff * omega**2 * (X**2 + Y**2)) / e_c

    # 2. Solve 2D Schrödinger equation
    solver = SchrodingerSolver2D(m_star=m_eff)
    evals_eV, psi_list = solver.solve(V_eV, x, y, num_states=6)
    evals_meV = evals_eV * 1e3

    # 3. Analytical energies E_(nx,ny) = ℏω (nx + ny + 1)
    analytical_meV = np.array([
        hbar_omega_meV * 1.0,  # (0,0)
        hbar_omega_meV * 2.0,  # (1,0)
        hbar_omega_meV * 2.0,  # (0,1)
        hbar_omega_meV * 3.0,  # (2,0)
        hbar_omega_meV * 3.0,  # (1,1)
        hbar_omega_meV * 3.0,  # (0,2)
    ])

    # 4. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp01_energy_levels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["State_Index", "Numerical_meV", "Analytical_meV", "Abs_Error_meV", "Rel_Error_percent"])
        for idx in range(min(len(evals_meV), len(analytical_meV))):
            num  = evals_meV[idx]
            ana  = analytical_meV[idx]
            err  = abs(num - ana)
            rel  = (err / ana) * 100
            writer.writerow([idx, num, ana, err, rel])
    print(f"Saved: {csv_path}")

    # 5. Save Metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp01_metadata.json")
    meta = {
        "experiment": "01_harmonic",
        "hbar_omega_meV": hbar_omega_meV,
        "grid_points": N,
        "grid_size_nm": L * 1e9,
        "ground_state_error_percent": float(abs(evals_meV[0] - analytical_meV[0]) / analytical_meV[0] * 100)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 6. Plot publication figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    # Ground state wave function density |ψ0|^2
    density0 = np.abs(psi_list[0])**2
    im0 = axes[0].pcolormesh(x*1e9, y*1e9, density0, cmap='viridis', shading='auto')
    fig.colorbar(im0, ax=axes[0], label=r'$|\psi_0(x,y)|^2$')
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('y (nm)')
    axes[0].set_title('Ground State Probability Density')

    # First excited state |ψ1|^2
    density1 = np.abs(psi_list[1])**2
    im1 = axes[1].pcolormesh(x*1e9, y*1e9, density1, cmap='magma', shading='auto')
    fig.colorbar(im1, ax=axes[1], label=r'$|\psi_1(x,y)|^2$')
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('y (nm)')
    axes[1].set_title('1st Excited State Density')

    # Energy comparison bar chart
    states = [f'E_{i}' for i in range(len(evals_meV))]
    x_pos = np.arange(len(states))
    width = 0.35
    axes[2].bar(x_pos - width/2, evals_meV, width, label='Numerical FD', color='royalblue')
    axes[2].bar(x_pos + width/2, analytical_meV, width, label='Analytical', color='darkorange')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(states)
    axes[2].set_ylabel('Energy (meV)')
    axes[2].set_title('Energy Spectrum Comparison')
    axes[2].legend()

    img_base = os.path.join(RESULTS_DIR, "exp01_wavefunction")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")
    print(f"E0 numerical = {evals_meV[0]:.4f} meV vs analytical = {analytical_meV[0]:.4f} meV")

if __name__ == '__main__':
    run()
