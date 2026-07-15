"""
Experiment 03: Double Quantum Dot Characterization
===================================================
Simulates charging energies, singlet-triplet exchange splitting J(ε),
and double dot stability cross-sections.

Saves:
  - QuantumTwin/results/exp03_double_dot.png (.pdf)
  - QuantumTwin/results/exp03_double_dot_data.csv
  - QuantumTwin/results/exp03_metadata.json
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
from physics.hubbard import DoubleDotST

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 03: Double Quantum Dot ---")
    np.random.seed(42)

    # Parameters
    U   = 4.0e-3    # 4.0 meV
    U12 = 0.8e-3    # 0.8 meV
    alpha = 0.12    # lever arm
    t0  = 50e-6     # 50 µeV tunnel coupling

    # 1. Constant Interaction Charge Stability
    U_onsite = np.array([U, U])
    U_inter  = np.array([[0, U12], [U12, 0]])
    lever_arms = np.array([[alpha, 0], [0, alpha]])

    csd = ChargeStabilityDiagram(U_onsite, U_inter, lever_arms, T=0.1, max_occupation=2)
    Vg_x = np.linspace(0.00, 0.05, 35)
    Vg_y = np.linspace(0.00, 0.05, 35)
    csd_res = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=True)

    # 2. Hubbard model exchange splitting J vs detuning
    st = DoubleDotST(U1=U, U2=U, U12=U12, t0=t0)
    eps_range = np.linspace(-3.0e-3, 3.0e-3, 100)
    sweep_hub = st.sweep_detuning(eps_range)

    # 3. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp03_double_dot_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Detuning_meV", "E_Singlet0_meV", "E_Triplet0_meV", "Exchange_J_meV", "Exchange_J_ueV"])
        for idx in range(len(eps_range)):
            eps_m = eps_range[idx] * 1e3
            e_s0  = sweep_hub['E_singlets'][idx, 0] * 1e3
            e_t0  = sweep_hub['E_T0'][idx] * 1e3
            j_m   = sweep_hub['J'][idx] * 1e3
            j_u   = sweep_hub['J'][idx] * 1e6
            writer.writerow([eps_m, e_s0, e_t0, j_m, j_u])
    print(f"Saved: {csv_path}")

    # 4. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp03_metadata.json")
    meta = {
        "experiment": "03_double_dot",
        "U_meV": 4.0,
        "U12_meV": 0.8,
        "tunnel_coupling_ueV": 50.0,
        "lever_arm": alpha,
        "peak_exchange_J_ueV": float(np.max(sweep_hub['J']) * 1e6),
        "triple_points_count": len(csd_res['triple_points'])
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 5. Plot figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Charge stability diagram
    im0 = axes[0].pcolormesh(Vg_x*1e3, Vg_y*1e3, csd_res['charge_map'], cmap='Blues', shading='nearest')
    fig.colorbar(im0, ax=axes[0], label='Total Charge N')
    if csd_res['triple_points']:
        tp = np.array(csd_res['triple_points']) * 1e3
        axes[0].scatter(tp[:, 0], tp[:, 1], c='red', s=20, label='Triple points')
        axes[0].legend(fontsize=9)
    axes[0].set_xlabel(r'$V_{g1}$ (mV)')
    axes[0].set_ylabel(r'$V_{g2}$ (mV)')
    axes[0].set_title('Double Dot Charge Stability')

    # Exchange splitting J(ε)
    axes[1].plot(eps_range*1e3, sweep_hub['J']*1e6, 'indigo', lw=2)
    axes[1].set_xlabel(r'Detuning $\varepsilon$ (meV)')
    axes[1].set_ylabel(r'Exchange Splitting $J$ ($\mu$eV)')
    axes[1].set_title(r'Singlet-Triplet Exchange $J(\varepsilon)$')
    axes[1].set_ylim(bottom=0)

    img_base = os.path.join(RESULTS_DIR, "exp03_double_dot")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")

if __name__ == '__main__':
    run()
