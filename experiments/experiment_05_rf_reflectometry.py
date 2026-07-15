"""
Experiment 05: RF Reflectometry Simulation
===========================================
Simulates LC resonator circuit profiles, quantum capacitance C_q, reflection
coefficient phase shifts, and demodulated IQ signals.

Saves:
  - QuantumTwin/results/exp05_rf_reflectometry.png (.pdf)
  - QuantumTwin/results/exp05_rf_data.csv
  - QuantumTwin/results/exp05_metadata.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from physics.reflectometry import RFReflectometrySimulator
from physics.hubbard import DoubleDotST

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 05: RF Reflectometry ---")
    np.random.seed(42)

    # 1. Setup RF resonator simulator
    sim = RFReflectometrySimulator(L_p=120e-9, C_p=1.0e-12, Q=80.0, Z_0=50.0, V_in=1e-3)
    f0  = sim.f_0

    # 2. Setup energy curve across charge transition
    t     = 15e-6   # 15 µeV tunnel coupling
    alpha = 0.12    # lever arm
    dV    = 20e-6   # 20 µV perturbation step

    eps_eV  = np.linspace(-0.3e-3, 0.3e-3, 200) # -0.3 to +0.3 meV
    eps_meV = eps_eV * 1e3

    # E0(eps) = -0.5 * sqrt(eps^2 + 4t^2) in eV
    E0 = lambda eps: -0.5 * np.sqrt(eps**2 + 4.0 * t**2)

    # 3. Compute quantum capacitance, reflection, and IQ signals
    Cq_list    = []
    phase_list = []
    I_list     = []
    Q_list     = []

    Gamma_ref, _, _ = sim.get_reflection(f0, C_q=0.0)

    for eps in eps_eV:
        # Compute C_q via finite difference
        Cq = sim.compute_quantum_capacitance(
            E0(eps - alpha*dV),
            E0(eps),
            E0(eps + alpha*dV),
            dV=dV
        )
        Cq_list.append(Cq)

        Gamma_test, amp, phase = sim.get_reflection(f0, C_q=Cq)
        I, Q, dtheta = sim.get_demodulated_iq(Gamma_test, Gamma_ref)

        phase_list.append(np.degrees(dtheta))
        I_list.append(I * 1e3) # mV
        Q_list.append(Q * 1e3) # mV

    Cq_arr    = np.array(Cq_list)
    phase_arr = np.array(phase_list)
    I_arr     = np.array(I_list)
    Q_arr     = np.array(Q_list)

    # 4. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp05_rf_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Detuning_meV", "Cq_fF", "PhaseShift_deg", "I_mV", "Q_mV"])
        for idx in range(len(eps_meV)):
            writer.writerow([eps_meV[idx], Cq_arr[idx]*1e15, phase_arr[idx], I_arr[idx], Q_arr[idx]])
    print(f"Saved: {csv_path}")

    # 5. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp05_metadata.json")
    meta = {
        "experiment": "05_rf_reflectometry",
        "resonance_frequency_MHz": float(f0 / 1e6),
        "peak_Cq_fF": float(np.max(Cq_arr) * 1e15),
        "max_phase_shift_deg": float(np.max(np.abs(phase_arr))),
        "IQ_loop_radius_mV": float(np.std(I_arr))
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 6. Plot figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # C_q vs detuning
    axes[0].plot(eps_meV, Cq_arr * 1e15, 'crimson', lw=2)
    axes[0].set_xlabel(r'Detuning $\varepsilon$ (meV)')
    axes[0].set_ylabel(r'Quantum Capacitance $C_q$ (fF)')
    axes[0].set_title(r'$C_q$ vs Detuning')

    # Phase shift vs detuning
    axes[1].plot(eps_meV, phase_arr, 'indigo', lw=2)
    axes[1].set_xlabel(r'Detuning $\varepsilon$ (meV)')
    axes[1].set_ylabel(r'Phase Shift $\Delta\theta$ (deg)')
    axes[1].set_title('RF Phase Shift')

    # IQ loop
    sc = axes[2].scatter(I_arr, Q_arr, c=eps_meV, cmap='RdBu', s=12)
    fig.colorbar(sc, ax=axes[2], label=r'$\varepsilon$ (meV)')
    axes[2].set_xlabel('I (mV)')
    axes[2].set_ylabel('Q (mV)')
    axes[2].set_title('Demodulated IQ Loop')
    axes[2].set_aspect('equal')

    img_base = os.path.join(RESULTS_DIR, "exp05_rf_reflectometry")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")

if __name__ == '__main__':
    run()
