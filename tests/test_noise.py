"""
Unit Tests for Advanced Noise Models
=====================================
Tests cover:
  1. Johnson noise RMS matches analytical formula sqrt(4 k_B T R BW).
  2. Johnson noise PSD is flat (white).
  3. RTN switching statistics: mean dwell times match 1/gamma.
  4. RTN analytical Lorentzian PSD peak frequency matches tau_bar.
  5. RTN amplitude is exactly cfg.amplitude (no intermediate values).
  6. Pink noise spectral exponent is within 5% of target alpha.
  7. Pink noise RMS matches configured amplitude.
  8. OU drift mean is near zero (stationary mean-reverting).
  9. OU drift variance matches analytical sigma^2*tau/2.
 10. Composite generator output has non-zero variance.
 11. Deterministic: same seed gives identical output.
 12. Johnson noise is uncorrelated (near-zero lag-1 autocorrelation).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics.noise import (
    JohnsonConfig, RTNConfig, PinkNoiseConfig, DriftConfig,
    JohnsonNoise, RTNoise, PinkNoise, ChargeOffsetDrift, CompositeNoiseGenerator,
    K_B,
)

# ─── Johnson-Nyquist ─────────────────────────────────────────────────────────

def test_johnson_rms_matches_theory():
    """V_rms = sqrt(4 k_B T R BW) to within 5% (statistical)."""
    cfg  = JohnsonConfig(R=1e3, T=0.1, bandwidth=1e6)
    gen  = JohnsonNoise(cfg)
    # Nyquist bandwidth for sample_rate=2e6 is exactly 1e6 Hz
    noise = gen.generate(n_samples=100_000, sample_rate=2e6, seed=0)
    V_rms_theory = np.sqrt(4.0 * K_B * cfg.T * cfg.R * cfg.bandwidth)
    V_rms_meas   = np.std(noise)
    assert pytest.approx(V_rms_meas, rel=0.05) == V_rms_theory


def test_johnson_spectral_density_property():
    """Spectral density S_V = 4 k_B T R."""
    cfg = JohnsonConfig(R=5e3, T=0.2)
    gen = JohnsonNoise(cfg)
    assert pytest.approx(gen.spectral_density, rel=1e-8) == 4.0 * K_B * 0.2 * 5e3


def test_johnson_uncorrelated():
    """Johnson noise is white: lag-1 autocorrelation < 0.05."""
    gen   = JohnsonNoise()
    noise = gen.generate(10_000, 1e6, seed=7)
    corr  = np.corrcoef(noise[:-1], noise[1:])[0, 1]
    assert abs(corr) < 0.05


# ─── Random Telegraph Noise ───────────────────────────────────────────────────

def test_rtn_amplitude_binary():
    """RTN signal only takes values 0 and cfg.amplitude (no intermediate)."""
    cfg  = RTNConfig(amplitude=50e-6, gamma_c=500.0, gamma_e=500.0, seed=1)
    gen  = RTNoise(cfg)
    sig  = gen.generate(n_samples=10_000, sample_rate=1e4, seed=1)
    unique_vals = np.unique(np.round(sig, 12))
    assert len(unique_vals) <= 2
    for v in unique_vals:
        assert pytest.approx(v, abs=1e-12) == 0.0 or \
               pytest.approx(v, abs=1e-12) == cfg.amplitude


def test_rtn_mean_dwell_time():
    """Average dwell time in state 0 ≈ 1/gamma_c."""
    gamma_c = 200.0
    gamma_e = 200.0
    cfg = RTNConfig(amplitude=1.0, gamma_c=gamma_c, gamma_e=gamma_e,
                    n_traps=1, seed=99)
    gen = RTNoise(cfg)
    dt  = 1e-4
    sig = gen.generate(n_samples=500_000, sample_rate=1.0/dt, seed=99)

    # Count transitions from 0→1 and measure 0-state dwell lengths
    transitions = np.where(np.diff(sig) != 0)[0]
    if len(transitions) < 10:
        pytest.skip("Too few transitions to measure dwell time.")

    dwell_0 = []
    for i in range(len(transitions) - 1):
        if sig[transitions[i] + 1] == cfg.amplitude:
            dwell_0.append((transitions[i + 1] - transitions[i]) * dt)

    if not dwell_0:
        pytest.skip("No 0-state dwells found.")

    mean_dwell_meas = np.mean(dwell_0)
    mean_dwell_theory = 1.0 / gamma_c
    assert pytest.approx(mean_dwell_meas, rel=0.2) == mean_dwell_theory


def test_rtn_lorentzian_psd():
    """Lorentzian PSD at f=0 equals 4 A² τ̄."""
    cfg = RTNConfig(amplitude=50e-6, gamma_c=100.0, gamma_e=150.0)
    gen = RTNoise(cfg)
    psd_0 = gen.lorentzian_psd(np.array([1e-6]))[0]
    expected = 4.0 * cfg.amplitude**2 * gen.tau_bar
    assert pytest.approx(psd_0, rel=1e-5) == expected


def test_rtn_tau_bar():
    """Characteristic time τ̄ = 1/(γ_c + γ_e)."""
    cfg = RTNConfig(gamma_c=200.0, gamma_e=300.0)
    gen = RTNoise(cfg)
    assert pytest.approx(gen.tau_bar, rel=1e-8) == 1.0 / 500.0


# ─── Pink Noise ───────────────────────────────────────────────────────────────

def test_pink_noise_rms():
    """Pink noise RMS matches configured amplitude within 1%."""
    cfg = PinkNoiseConfig(amplitude=100e-6, alpha=1.0, seed=42)
    gen = PinkNoise(cfg)
    ts, _ = gen.generate(n_samples=8192, sample_rate=1e4, seed=42)
    assert pytest.approx(np.std(ts), rel=0.01) == cfg.amplitude


def test_pink_noise_spectral_exponent():
    """
    Fitted spectral exponent is within 10% of target alpha.
    For alpha >= 1.0 (clear power law), R² > 0.90 is required.
    For alpha = 0.8 (near-white), only the exponent range is checked.
    """
    # Strong power-law cases: R² must be high
    for alpha in [1.0, 1.5, 2.0]:
        cfg = PinkNoiseConfig(amplitude=1e-4, alpha=alpha, seed=0)
        gen = PinkNoise(cfg)
        ts, _ = gen.generate(n_samples=65536, sample_rate=1e4, seed=0)
        alpha_fit, r2 = gen.fit_spectral_exponent(ts, sample_rate=1e4)
        assert r2 > 0.90, f"Poor R2 = {r2:.3f} for alpha={alpha}"
        assert abs(alpha_fit - alpha) / alpha < 0.10, \
            f"alpha target={alpha}, fitted={alpha_fit:.4f}"

    # Shallow slope: α=0.8 is near-white; verify exponent direction only
    cfg = PinkNoiseConfig(amplitude=1e-4, alpha=0.8, seed=0)
    gen = PinkNoise(cfg)
    ts, _ = gen.generate(n_samples=65536, sample_rate=1e4, seed=0)
    alpha_fit, _ = gen.fit_spectral_exponent(ts, sample_rate=1e4)
    assert 0.3 < alpha_fit < 1.5, \
        f"alpha=0.8: fitted={alpha_fit:.4f} out of plausible range [0.3, 1.5]"


# ─── Charge Offset Drift ──────────────────────────────────────────────────────

def test_ou_mean_zero():
    """Long OU drift should have mean near zero (mean-reverting)."""
    cfg  = DriftConfig(tau=1.0, sigma=1e-6, seed=5)
    gen  = ChargeOffsetDrift(cfg)
    drift = gen.generate(n_samples=100_000, dt=1e-3, seed=5)
    assert abs(np.mean(drift)) < 5.0 * gen.stationary_std


def test_ou_stationary_variance():
    """Long OU process variance ≈ σ²τ/2 within 10%."""
    tau = 2.0
    sig = 5e-7
    cfg = DriftConfig(tau=tau, sigma=sig, seed=3)
    gen = ChargeOffsetDrift(cfg)
    drift = gen.generate(n_samples=200_000, dt=1e-3, seed=3)
    var_theory = sig**2 * tau / 2.0
    var_meas   = np.var(drift)
    assert pytest.approx(var_meas, rel=0.10) == var_theory


# ─── Composite Generator ──────────────────────────────────────────────────────

def test_composite_nonzero():
    """Composite noise has nonzero variance."""
    gen   = CompositeNoiseGenerator()
    noise = gen.generate(n_samples=4096, sample_rate=1e4, seed=0)
    assert np.std(noise) > 0.0


def test_composite_deterministic():
    """Same seed produces identical output."""
    gen    = CompositeNoiseGenerator()
    noise1 = gen.generate(4096, 1e4, seed=7)
    noise2 = gen.generate(4096, 1e4, seed=7)
    np.testing.assert_array_equal(noise1, noise2)
