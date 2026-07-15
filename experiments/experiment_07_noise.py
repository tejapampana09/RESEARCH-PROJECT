"""
Experiment 07: Noise Models Characterization
=============================================
Simulates Johnson-Nyquist thermal noise, Random Telegraph Noise (Lorentzian PSD),
1/f pink charge noise, and Ornstein-Uhlenbeck charge drift.

Saves:
  - QuantumTwin/results/exp07_noise.png (.pdf)
  - QuantumTwin/results/exp07_noise_data.csv
  - QuantumTwin/results/exp07_metadata.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import json

from physics.noise import (
    JohnsonConfig, RTNConfig, PinkNoiseConfig, DriftConfig,
    JohnsonNoise, RTNoise, PinkNoise, ChargeOffsetDrift,
    CompositeNoiseGenerator
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run():
    print("--- Running Experiment 07: Advanced Noise Models ---")
    np.random.seed(42)

    N_samples   = 32768
    sample_rate = 10000.0   # 10 kHz

    # 1. Instantiate noise generators
    johnson = JohnsonNoise(JohnsonConfig(R=1e3, T=0.1, bandwidth=sample_rate/2))
    rtn     = RTNoise(RTNConfig(amplitude=50e-6, gamma_c=80.0, gamma_e=120.0, seed=42))
    pink    = PinkNoise(PinkNoiseConfig(amplitude=100e-6, alpha=1.0, seed=42))
    drift   = ChargeOffsetDrift(DriftConfig(tau=5.0, sigma=5e-7, seed=42))

    # 2. Generate time series
    j_ts    = johnson.generate(N_samples, sample_rate, seed=1)
    r_ts    = rtn.generate(N_samples, sample_rate, seed=2)
    p_ts, _ = pink.generate(N_samples, sample_rate, seed=3)
    d_ts    = drift.generate(N_samples, dt=1.0/sample_rate, seed=4)

    # 3. Spectral exponent fit for pink noise using Welch PSD
    alpha_fit, r2 = pink.fit_spectral_exponent(p_ts, sample_rate)

    # 4. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp07_noise_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise_Type", "RMS_amplitude_uV", "Theoretical_metric"])
        writer.writerow(["Johnson", float(np.std(j_ts)*1e6), float(johnson.rms_voltage*1e6)])
        writer.writerow(["RTN", float(np.std(r_ts)*1e6), float(rtn.cfg.amplitude*1e6)])
        writer.writerow(["Pink_1f", float(np.std(p_ts)*1e6), float(pink.cfg.amplitude*1e6)])
        writer.writerow(["OU_Drift", float(np.std(d_ts)*1e6), float(drift.stationary_std*1e6)])
    print(f"Saved: {csv_path}")

    # 5. Save metadata JSON
    meta_path = os.path.join(RESULTS_DIR, "exp07_metadata.json")
    meta = {
        "experiment": "07_noise",
        "sample_rate_Hz": sample_rate,
        "n_samples": N_samples,
        "pink_target_alpha": 1.0,
        "pink_fitted_alpha": float(alpha_fit),
        "pink_R2": float(r2),
        "johnson_rms_measured_uV": float(np.std(j_ts)*1e6),
        "johnson_rms_theory_uV": float(johnson.rms_voltage*1e6)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # 6. Plot figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    t_ms = (np.arange(1000) / sample_rate) * 1e3
    axes[0].plot(t_ms, r_ts[:1000]*1e6, 'crimson', lw=1, label='RTN (Telegraph)')
    axes[0].plot(t_ms, p_ts[:1000]*1e6, 'darkorange', lw=1, alpha=0.8, label='1/f Pink Noise')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(r'Voltage ($\mu$V)')
    axes[0].set_title('Time-Domain Noise Signatures')
    axes[0].legend()

    from scipy.signal import welch
    f_p, psd_p = welch(p_ts, fs=sample_rate, nperseg=2048)
    axes[1].loglog(f_p[1:], psd_p[1:], 'darkorange', lw=1.5, label='1/f Welch PSD')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel(r'PSD ($V^2$/Hz)')
    axes[1].set_title(rf'1/f Power Spectrum ($\alpha_{{fit}}={alpha_fit:.3f}$, $R^2={r2:.3f}$)')
    axes[1].legend()

    img_base = os.path.join(RESULTS_DIR, "exp07_noise")
    for fmt in ('png', 'pdf'):
        fig.savefig(f"{img_base}.{fmt}", dpi=300)
    plt.close(fig)
    print(f"Saved: {img_base}.png/.pdf")

if __name__ == '__main__':
    run()
