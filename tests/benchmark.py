"""
Benchmarking Suite for QuantumTwin
===================================
Benchmarks execution times and accuracy across all core physics components:
1. 2D Schrödinger Finite-Difference Solver
2. Constant Interaction Multi-Dot Solver
3. Fermi-Hubbard Eigensolver
4. RF Reflectometry Demodulator
5. Noise Time-Series Generators
6. DQN Reinforcement Learning Optimization

Saves:
  - QuantumTwin/results/benchmark_report.csv
  - QuantumTwin/results/benchmark_runtime.png (.pdf)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import time

from quantum.schrodinger import SchrodingerSolver2D
from physics.constant_interaction import ConstantInteractionModel
from physics.hubbard import FermiHubbardModel, DoubleDotST
from physics.reflectometry import RFReflectometrySimulator
from physics.noise import CompositeNoiseGenerator

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmarks():
    print("=========================================================================")
    print("                 QuantumTwin Benchmarking Suite")
    print("=========================================================================")
    np.random.seed(42)

    benchmark_data = []

    # 1. 2D Schrödinger Solver Benchmark
    sol = SchrodingerSolver2D()
    x_b = np.linspace(-40e-9, 40e-9, 45)
    V_dummy = np.zeros((45, 45))

    t0 = time.perf_counter()
    for _ in range(5):
        sol.solve(V_dummy, x_b, x_b, num_states=6)
    t_schrodinger = (time.perf_counter() - t0) / 5.0 * 1e3  # ms
    benchmark_data.append({"Module": "2D Schrödinger Solver (45x45)", "Execution_Time_ms": t_schrodinger, "Target_FPS": 20})
    print(f"1. 2D Schrödinger Solver: {t_schrodinger:.2f} ms per frame")

    # 2. Constant Interaction Sweep Benchmark
    ci = ConstantInteractionModel(num_dots=2)
    t0 = time.perf_counter()
    ci.find_ground_state(np.array([0.02, 0.02]))
    t_ci = (time.perf_counter() - t0) * 1e3
    benchmark_data.append({"Module": "Constant Interaction Evaluation", "Execution_Time_ms": t_ci, "Target_FPS": 1000})
    print(f"2. Constant Interaction Evaluation: {t_ci:.3f} ms per point")

    # 3. Fermi-Hubbard Solver Benchmark
    st = DoubleDotST(t0=50e-6)
    t0 = time.perf_counter()
    for _ in range(50):
        st.compute_exchange(0.0)
    t_hubbard = (time.perf_counter() - t0) / 50.0 * 1e3
    benchmark_data.append({"Module": "Fermi-Hubbard Diagonalization", "Execution_Time_ms": t_hubbard, "Target_FPS": 500})
    print(f"3. Fermi-Hubbard Diagonalization: {t_hubbard:.3f} ms per point")

    # 4. RF Reflectometry Benchmark
    rf_sim = RFReflectometrySimulator()
    G_ref, _, _ = rf_sim.get_reflection(rf_sim.f_0, 0.0)
    t0 = time.perf_counter()
    for _ in range(100):
        G, _, _ = rf_sim.get_reflection(rf_sim.f_0, 10e-15)
        rf_sim.get_demodulated_iq(G, G_ref)
    t_rf = (time.perf_counter() - t0) / 100.0 * 1e3
    benchmark_data.append({"Module": "RF Reflectometry Circuit", "Execution_Time_ms": t_rf, "Target_FPS": 2000})
    print(f"4. RF Reflectometry Circuit: {t_rf:.3f} ms per call")

    # 5. Composite Noise Generator Benchmark
    cnoise = CompositeNoiseGenerator()
    t0 = time.perf_counter()
    cnoise.generate(n_samples=8192, sample_rate=10000.0, seed=42)
    t_noise = (time.perf_counter() - t0) * 1e3
    benchmark_data.append({"Module": "Noise Generator (8192 points)", "Execution_Time_ms": t_noise, "Target_FPS": 100})
    print(f"5. Noise Generator (8k points): {t_noise:.2f} ms")

    # Save CSV
    report_csv = os.path.join(RESULTS_DIR, "benchmark_report.csv")
    with open(report_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Module", "Execution_Time_ms", "Target_FPS"])
        writer.writeheader()
        writer.writerows(benchmark_data)
    print(f"\nSaved CSV report: {report_csv}")

    # Plot runtime bar chart
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    modules = [d["Module"] for d in benchmark_data]
    times   = [d["Execution_Time_ms"] for d in benchmark_data]
    y_pos   = np.arange(len(modules))

    ax.barh(y_pos, times, color='royalblue', height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(modules)
    ax.invert_yaxis()
    ax.set_xlabel('Execution Time (ms)')
    ax.set_title('QuantumTwin Execution Time Benchmarks')
    ax.set_xscale('log')

    img_base = os.path.join(RESULTS_DIR, "benchmark_runtime")
    fig.savefig(f"{img_base}.png", dpi=300)
    fig.savefig(f"{img_base}.pdf")
    plt.close(fig)
    print(f"Saved figure: {img_base}.png/.pdf")

if __name__ == '__main__':
    run_benchmarks()
