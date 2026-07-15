"""
Experiment 06: Fermi-Hubbard Model Characterization
====================================================
Simulates Fock space diagonalization of the Fermi-Hubbard Hamiltonian,
singlet-triplet energy spectra, and exchange coupling J(t).

Saves:
  - QuantumTwin/results/exp06_hubbard.png (.pdf)
  - QuantumTwin/results/exp06_hubbard_data.csv
  - QuantumTwin/results/exp06_metadata.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from physics.hubbard import FermiHubbardModel, DoubleDotST

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 06: Fermi-Hubbard Model ---")
    np.random.seed(42)

    # Parameters
    U   = 4.0e-3
    U12 = 0.8e-3
    st  = DoubleDotST(U1=U, U2=U, U12=U12, t0=50e-6)

    # 1. J vs tunnel coupling t sweep
    t_vals = np.linspace(5e-6, 150e-6, 100)
    J_num  = []
    J_anal = []

    for t in t_vals:
        res = st.compute_exchange(0.0, V_b=np.log(t / st.t0) / st.gamma)
        J_num.append(res['J'])
        J_anal.append(st.analytical_J_deep_detuning(t))

    J_num  = np.array(J_num)
    J_anal = np.array(J_anal)

    # 2. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp06_hubbard_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["t_ueV", "J_numerical_ueV", "J_analytical_ueV", "Error_percent"])
        for idx in range(len(t_vals)):
            t_u = t_vals[idx] * 1e6
            jn  = J_num[idx] * 1e6
            ja  = J_anal[idx] * 1e6
            err = abs(jn - ja) / ja * 100
            writer.writerow([t_u, jn, ja, err])
    print(f"Saved: {csv_path}")

    # 3. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp06_metadata.json")
    meta = {
        "experiment": "06_hubbard",
        "hilbert_space_dim": 6,
        "U_meV": 4.0,
        "U12_meV": 0.8,
        "weak_t_10ueV_error_percent": float(abs(J_num[5] - J_anal[5]) / J_anal[5] * 100),
        "peak_J_ueV": float(np.max(J_num) * 1e6)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 4. Plot figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # J vs t curve
    axes[0].plot(t_vals*1e6, J_num*1e6, 'k-', lw=2, label='Numerical (Hubbard)')
    axes[0].plot(t_vals*1e6, J_anal*1e6, 'r--', lw=1.8, label=r'Analytical $4t^2/(U-U_{12})$')
    axes[0].set_xlabel(r'Tunnel coupling $t$ ($\mu$eV)')
    axes[0].set_ylabel(r'Exchange $J$ ($\mu$eV)')
    axes[0].set_title(r'Exchange $J$ vs Tunnel Coupling $t$')
    axes[0].legend()

    # J vs Barrier Voltage
    Vb_vals = np.linspace(-0.08, 0.04, 100)
    J_Vb    = np.array([st.compute_exchange(0.0, V_b=vb)['J'] for vb in Vb_vals])
    axes[1].plot(Vb_vals*1e3, J_Vb*1e6, 'indigo', lw=2)
    axes[1].set_xlabel(r'Barrier Gate Voltage $V_b$ (mV)')
    axes[1].set_ylabel(r'Exchange $J$ ($\mu$eV)')
    axes[1].set_title(r'Tunability $J(V_b)$')
    axes[1].set_yscale('log')

    img_base = os.path.join(RESULTS_DIR, "exp06_hubbard")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")

if __name__ == '__main__':
    run()
