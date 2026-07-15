"""
Master Publication Result Generator for QuantumTwin
===================================================
Runs all 8 experiments sequentially, exports CSV data and JSON metadata,
and generates the master 8-panel publication summary figure.

Outputs:
  - QuantumTwin/results/publication_summary.png (.pdf)
  - Individual experiment figures and data files
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import all experiments
from experiments import (
    experiment_01_harmonic,
    experiment_02_single_dot,
    experiment_03_double_dot,
    experiment_04_charge_stability,
    experiment_05_rf_reflectometry,
    experiment_06_hubbard,
    experiment_07_noise,
    experiment_08_rl_training,
)

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def run_all():
    print("=========================================================================")
    print("       QuantumTwin Master Publication Result Generator")
    print("=========================================================================")

    # 1. Run all experiments sequentially
    experiment_01_harmonic.run()
    experiment_02_single_dot.run()
    experiment_03_double_dot.run()
    experiment_04_charge_stability.run()
    experiment_05_rf_reflectometry.run()
    experiment_06_hubbard.run()
    experiment_07_noise.run()
    experiment_08_rl_training.run()

    print("\n--- Generating Master 8-Panel Publication Figure ---")
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "savefig.dpi": 300})

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    # Panel A: 2D Harmonic wavefunction
    from quantum.schrodinger import SchrodingerSolver2D
    m_eff = 0.19 * 9.1093837015e-31; omega = 1.5e12; e_c = 1.602176634e-19
    N = 35; L = 80e-9
    x = np.linspace(-L/2, L/2, N); X, Y = np.meshgrid(x, x)
    V_eV = (0.5 * m_eff * omega**2 * (X**2 + Y**2)) / e_c
    sol = SchrodingerSolver2D()
    evals_eV, psi_list = sol.solve(V_eV, x, x, num_states=1)
    psi0 = np.abs(psi_list[0])**2
    axes[0, 0].pcolormesh(x*1e9, x*1e9, psi0, cmap='viridis', shading='auto')
    axes[0, 0].set_title('(a) Ground State wavefunction |ψ0|²')
    axes[0, 0].set_xlabel('x (nm)'); axes[0, 0].set_ylabel('y (nm)')

    # Panel B: Potential well
    from physics.potential import PotentialLandscape
    pot = PotentialLandscape(grid_size=(35, 35), extent=(80e-9, 80e-9))
    pot.add_dot("D1", 0.0, 0.0, depth=80e-3, radius=12e-9)
    pot.add_gate("P1", 25e-9, 0.0, lever_arm=0.12, sigma=15e-9)
    V_map = pot.get_potential({"P1": 0.05})
    axes[0, 1].pcolormesh(pot.x*1e9, pot.y*1e9, V_map.T*1e3, cmap='viridis_r', shading='auto')
    axes[0, 1].set_title('(b) Potential Landscape V(x,y)')
    axes[0, 1].set_xlabel('x (nm)'); axes[0, 1].set_ylabel('y (nm)')

    # Panel C: Honeycomb stability diagram
    from physics.charge_stability import ChargeStabilityDiagram
    csd = ChargeStabilityDiagram(np.array([4e-3, 4e-3]), np.array([[0, 0.8e-3], [0.8e-3, 0]]),
                                 np.array([[0.12, 0], [0, 0.12]]), T=0.1)
    res_csd = csd.sweep(0, 1, np.linspace(0, 0.05, 30), np.linspace(0, 0.05, 30))
    axes[0, 2].pcolormesh(res_csd['Vg_x']*1e3, res_csd['Vg_y']*1e3, res_csd['charge_map'],
                          cmap='Blues', shading='nearest')
    axes[0, 2].set_title('(c) Honeycomb Charge Stability')
    axes[0, 2].set_xlabel('Vg1 (mV)'); axes[0, 2].set_ylabel('Vg2 (mV)')

    # Panel D: Singlet-Triplet exchange J(ε)
    from physics.hubbard import DoubleDotST
    st = DoubleDotST(t0=50e-6)
    eps_range = np.linspace(-3e-3, 3e-3, 60)
    sw = st.sweep_detuning(eps_range)
    axes[0, 3].plot(eps_range*1e3, sw['J']*1e6, 'indigo', lw=2)
    axes[0, 3].set_title('(d) Exchange Splitting J(ε)')
    axes[0, 3].set_xlabel('Detuning (meV)'); axes[0, 3].set_ylabel('J (µeV)')

    # Panel E: RF Reflectometry phase shift
    from physics.reflectometry import RFReflectometrySimulator
    sim = RFReflectometrySimulator()
    t = 15e-6; alpha = 0.12; dV = 20e-6
    eps_eV = np.linspace(-0.3e-3, 0.3e-3, 60)
    E0 = lambda e: -0.5 * np.sqrt(e**2 + 4.0 * t**2)
    G_ref, _, _ = sim.get_reflection(sim.f_0, 0.0)
    phases = []
    for e_val in eps_eV:
        Cq = sim.compute_quantum_capacitance(E0(e_val-alpha*dV), E0(e_val), E0(e_val+alpha*dV), dV=dV)
        G_t, _, _ = sim.get_reflection(sim.f_0, Cq)
        _, _, dth = sim.get_demodulated_iq(G_t, G_ref)
        phases.append(np.degrees(dth))
    axes[1, 0].plot(eps_eV*1e3, phases, 'crimson', lw=2)
    axes[1, 0].set_title('(e) RF Phase Shift Δθ')
    axes[1, 0].set_xlabel('Detuning (meV)'); axes[1, 0].set_ylabel('Δθ (deg)')

    # Panel F: Noise PSD spectrum
    from physics.noise import PinkNoise, PinkNoiseConfig
    from scipy.signal import welch
    pn = PinkNoise(PinkNoiseConfig(amplitude=100e-6, alpha=1.0, seed=42))
    p_ts, _ = pn.generate(8192, 1e4, seed=42)
    fp, pp = welch(p_ts, fs=1e4, nperseg=1024)
    axes[1, 1].loglog(fp[1:], pp[1:], 'darkorange', lw=1.5)
    axes[1, 1].set_title('(f) 1/f Pink Noise PSD')
    axes[1, 1].set_xlabel('Frequency (Hz)'); axes[1, 1].set_ylabel('PSD (V²/Hz)')

    # Panel G: Hubbard J vs t
    t_vals = np.linspace(5e-6, 120e-6, 50)
    J_num = [st.compute_exchange(0.0, V_b=np.log(tv/st.t0)/st.gamma)['J']*1e6 for tv in t_vals]
    axes[1, 2].plot(t_vals*1e6, J_num, 'k-', lw=2)
    axes[1, 2].set_title('(g) Exchange J vs Coupling t')
    axes[1, 2].set_xlabel('t (µeV)'); axes[1, 2].set_ylabel('J (µeV)')

    # Panel H: DQN RL training curve
    from ai.tuning import DQNTuningAgent
    from simulator.device import SiliconQDArray
    dev_rl = SiliconQDArray(num_dots=2)
    agent_rl = DQNTuningAgent(state_dim=2, action_dim=4, lr=1e-3)
    steps_list = []
    for _ in range(12):
        dev_rl.gate_voltages["P1"] = 0.01
        dev_rl.gate_voltages["P2"] = 0.01
        tr = agent_rl.tune_device(dev_rl, target_state=(1, 1), max_steps=20, dv=3.0e-3)
        steps_list.append(tr['steps'])
    axes[1, 3].plot(np.arange(1, 13), steps_list, 's-', color='royalblue', lw=2)
    axes[1, 3].set_title('(h) DQN Tuning Steps')
    axes[1, 3].set_xlabel('Episode'); axes[1, 3].set_ylabel('Steps to Target')

    img_master = os.path.join(RESULTS_DIR, "publication_summary")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_master}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"\nSaved master publication figure: {img_master}.png/.pdf")
    print("=========================================================================")
    print("               Master Publication Benchmark Completed!")
    print("=========================================================================")

if __name__ == '__main__':
    run_all()
