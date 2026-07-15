"""
Advanced Noise Models for Silicon Quantum Dot Arrays
=====================================================

Implements four physically distinct noise sources that affect
gate-defined quantum dot devices:

  1. Johnson-Nyquist Noise   — thermal white noise from resistive gate lines
  2. Random Telegraph Noise  — Poisson switching of individual charge traps
  3. 1/f Charge Noise        — spectral-shaped low-frequency charge fluctuations
  4. Charge Offset Drift     — slow Ornstein-Uhlenbeck drift of the charge offset

Physical Basis
--------------
  Johnson:  S_V(f) = 4 k_B T R  [V²/Hz]
  RTN PSD:  S_RTN(f) = 4 A² τ̄ / (1 + (2π f τ̄)²)   (Lorentzian)
  1/f PSD:  S(f) = A² / f^α
  OU drift: d(δng)/dt = -δng/τ + σ ξ(t)

References
----------
1. E. Paladino et al., "1/f noise: Implications for solid-state quantum information,"
   Rev. Mod. Phys. 86, 361 (2014).
2. T. Elo et al., "Low-frequency noise in silicon quantum dot qubits,"
   Phys. Rev. Applied 19, 034044 (2023).
3. J. B. Johnson, "Thermal agitation of electricity in conductors,"
   Phys. Rev. 32, 97 (1928).
4. H. Nyquist, "Thermal agitation of electric charge in conductors,"
   Phys. Rev. 32, 110 (1928).
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Physical constants
K_B   = 1.380649e-23    # Boltzmann constant (J/K)
E_CHG = 1.602176634e-19 # Electron charge (C)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes for Noise Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JohnsonConfig:
    """Configuration for Johnson-Nyquist noise.

    Attributes:
        R      : Gate line resistance (Ohm). Typical RC filter: 1–10 kOhm.
        T      : Electron temperature (K). Typical dilution fridge: 0.05–0.3 K.
        bandwidth: Measurement bandwidth (Hz). Defines noise floor.
    """
    R: float = 1e3       # 1 kOhm
    T: float = 0.1       # 100 mK
    bandwidth: float = 1e6  # 1 MHz


@dataclass
class RTNConfig:
    """Configuration for Random Telegraph Noise.

    A single two-level charge trap with capture rate γ_c and emission rate γ_e.

    Attributes:
        amplitude  : Voltage amplitude of each telegraph jump (V).
        gamma_c    : Capture rate γ_c (Hz). Mean dwell time in 0-state = 1/γ_c.
        gamma_e    : Emission rate γ_e (Hz). Mean dwell time in 1-state = 1/γ_e.
        n_traps    : Number of independent RTN traps to superpose.
        seed       : Random seed for reproducibility.
    """
    amplitude: float  = 50e-6    # 50 µV switching amplitude
    gamma_c  : float  = 100.0    # 100 Hz capture rate
    gamma_e  : float  = 150.0    # 150 Hz emission rate
    n_traps  : int    = 1
    seed     : Optional[int] = 42


@dataclass
class PinkNoiseConfig:
    """Configuration for 1/f^α charge noise.

    Attributes:
        amplitude: RMS amplitude of noise (V).
        alpha    : Spectral exponent. α=1 for pure 1/f, α=2 for Brownian.
        seed     : Random seed for reproducibility.
    """
    amplitude: float = 100e-6   # 100 µV RMS
    alpha    : float = 1.0
    seed     : Optional[int] = 42


@dataclass
class DriftConfig:
    """Configuration for Ornstein-Uhlenbeck charge offset drift.

    Attributes:
        tau        : Correlation time of the drift (s). Typical: 1–100 s.
        sigma      : Diffusion coefficient (V/sqrt(s)).
        initial    : Initial offset value (V).
        seed       : Random seed for reproducibility.
    """
    tau    : float = 10.0        # 10 s correlation time
    sigma  : float = 1e-6        # 1 µV/sqrt(s) diffusion
    initial: float = 0.0
    seed   : Optional[int] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Johnson-Nyquist Noise
# ─────────────────────────────────────────────────────────────────────────────

class JohnsonNoise:
    """
    Generates Johnson-Nyquist (thermal) white noise time series.

    The voltage noise spectral density across a resistor R at temperature T is:
        S_V(f) = 4 k_B T R  [V²/Hz]

    The RMS amplitude over bandwidth B is:
        V_rms = sqrt(4 k_B T R B)

    Time-domain samples are i.i.d. Gaussian with std = V_rms / sqrt(N)
    where N is the number of samples drawn per unit time.
    """

    def __init__(self, config: Optional[JohnsonConfig] = None):
        self.cfg = config or JohnsonConfig()

    @property
    def spectral_density(self) -> float:
        """Single-sided PSD S_V(f) = 4 k_B T R  [V²/Hz]."""
        return 4.0 * K_B * self.cfg.T * self.cfg.R

    @property
    def rms_voltage(self) -> float:
        """RMS voltage over the configured bandwidth [V]."""
        return np.sqrt(self.spectral_density * self.cfg.bandwidth)

    def generate(self, n_samples: int, sample_rate: float,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a white Gaussian noise time series.

        Args:
            n_samples   : Number of time steps.
            sample_rate : Sampling rate f_s (Hz). Nyquist bandwidth = f_s/2.
            seed        : Optional random seed.

        Returns:
            noise: Array of shape (n_samples,) in Volts.
        """
        rng = np.random.default_rng(seed)
        # Effective bandwidth = Nyquist limit min(bandwidth, f_s/2)
        bw   = min(self.cfg.bandwidth, sample_rate / 2.0)
        vrms = np.sqrt(4.0 * K_B * self.cfg.T * self.cfg.R * bw)
        noise = rng.normal(loc=0.0, scale=vrms, size=n_samples)
        logger.debug(f"JohnsonNoise: V_rms = {vrms*1e6:.4f} µV over {bw/1e3:.1f} kHz BW")
        return noise


# ─────────────────────────────────────────────────────────────────────────────
# Random Telegraph Noise (RTN)
# ─────────────────────────────────────────────────────────────────────────────

class RTNoise:
    """
    Generates Random Telegraph Noise (RTN) from Poisson-distributed charge traps.

    Each trap is a two-level fluctuator switching between V=0 and V=amplitude
    via exponentially distributed dwell times.

    The power spectral density of a single RTN source is Lorentzian:
        S_RTN(f) = 4 A² τ̄ / (1 + (2π f τ̄)²)
    where:
        τ̄  = 1 / (γ_c + γ_e)   : characteristic switching time
        A   : amplitude of the jump

    Multiple independent traps are superposed to approximate 1/f noise
    (Dutta-Horn model: sum of Lorentzians with distributed τ).
    """

    def __init__(self, config: Optional[RTNConfig] = None):
        self.cfg = config or RTNConfig()

    @property
    def tau_bar(self) -> float:
        """Characteristic switching time τ̄ = 1/(γ_c + γ_e)."""
        return 1.0 / (self.cfg.gamma_c + self.cfg.gamma_e)

    def lorentzian_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Returns the analytical Lorentzian PSD of this RTN source.

        S_RTN(f) = 4 A² τ̄ / (1 + (2π f τ̄)²)

        Args:
            frequencies: Array of frequencies (Hz).

        Returns:
            PSD array in V²/Hz.
        """
        A   = self.cfg.amplitude
        tau = self.tau_bar
        return 4.0 * A**2 * tau / (1.0 + (2.0 * np.pi * frequencies * tau)**2)

    def _generate_single_trap(self, duration: float, dt: float,
                               rng: np.random.Generator) -> np.ndarray:
        """
        Generates a single RTN time series using the Poisson switching process.

        Args:
            duration : Total simulation time (s).
            dt       : Time step (s).
            rng      : NumPy random generator.

        Returns:
            Voltage time series (V) of length ceil(duration/dt).
        """
        n_samples = int(np.ceil(duration / dt))
        signal    = np.zeros(n_samples)

        # Steady-state probability of being in state 1:
        p1 = self.cfg.gamma_c / (self.cfg.gamma_c + self.cfg.gamma_e)
        state = int(rng.random() < p1)   # Initial state (0 or 1)

        t = 0.0
        while t < duration:
            # Sample dwell time in current state
            rate = self.cfg.gamma_c if state == 0 else self.cfg.gamma_e
            dwell = rng.exponential(1.0 / rate)

            # Fill signal with current state value
            i_start = int(t / dt)
            i_end   = min(int((t + dwell) / dt), n_samples)
            signal[i_start:i_end] = state * self.cfg.amplitude

            t    += dwell
            state = 1 - state   # Toggle state

        return signal

    def generate(self, n_samples: int, sample_rate: float,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generates RTN time series (sum of n_traps independent traps).

        Args:
            n_samples   : Number of time steps.
            sample_rate : Sampling rate (Hz).
            seed        : Optional random seed.

        Returns:
            noise: Array of shape (n_samples,) in Volts.
        """
        rng      = np.random.default_rng(seed if seed is not None else self.cfg.seed)
        dt       = 1.0 / sample_rate
        duration = n_samples * dt
        total    = np.zeros(n_samples)

        for k in range(self.cfg.n_traps):
            trap_rng = np.random.default_rng(rng.integers(0, 2**32))
            total   += self._generate_single_trap(duration, dt, trap_rng)

        logger.debug(f"RTNoise: {self.cfg.n_traps} trap(s), τ̄ = {self.tau_bar*1e3:.2f} ms, "
                     f"A = {self.cfg.amplitude*1e6:.1f} µV")
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 1/f Pink Noise (standalone, clean implementation)
# ─────────────────────────────────────────────────────────────────────────────

class PinkNoise:
    """
    Generates 1/f^α (pink) charge noise via inverse-FFT spectral shaping.

    Algorithm:
      1. Draw complex Gaussian amplitudes in the frequency domain.
      2. Scale by 1/f^(α/2) to obtain the correct PSD shape.
      3. Apply IFFT to obtain a real-valued time series.
      4. Normalise to the target RMS amplitude.

    The resulting time series has PSD ∝ 1/f^α with a measurable spectral
    exponent that matches α within numerical precision for long sequences.
    """

    def __init__(self, config: Optional[PinkNoiseConfig] = None):
        self.cfg = config or PinkNoiseConfig()

    def generate(self, n_samples: int, sample_rate: float,
                 seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 1/f^α noise.

        Args:
            n_samples   : Number of time steps (recommended: power of 2).
            sample_rate : Sampling rate f_s (Hz).
            seed        : Optional random seed.

        Returns:
            (time_series, frequencies):
                time_series : Real noise array (V), shape (n_samples,).
                frequencies : Positive frequency axis (Hz) for PSD analysis.
        """
        rng  = np.random.default_rng(seed if seed is not None else self.cfg.seed)
        N    = n_samples
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

        # Avoid DC divergence: set f=0 component to zero
        freqs[0] = 1.0   # temporary to avoid divide-by-zero

        # Spectral amplitude scaling: |X(f)| ∝ 1/f^(α/2)
        scaling = np.power(freqs, -self.cfg.alpha / 2.0)
        scaling[0] = 0.0   # zero DC component

        # Random complex amplitudes
        real_part = rng.standard_normal(len(freqs))
        imag_part = rng.standard_normal(len(freqs))
        X_f = (real_part + 1j * imag_part) * scaling

        # IFFT to time domain
        x_t = np.fft.irfft(X_f, n=N)

        # Normalise to target RMS amplitude
        rms_current = np.std(x_t)
        if rms_current > 1e-30:
            x_t = x_t / rms_current * self.cfg.amplitude

        logger.debug(f"PinkNoise: α={self.cfg.alpha}, RMS={np.std(x_t)*1e6:.2f} µV")
        # Restore real frequencies for output
        freqs[0] = 0.0
        return x_t, freqs

    def fit_spectral_exponent(self, x_t: np.ndarray,
                               sample_rate: float) -> Tuple[float, float]:
        """
        Estimates the spectral exponent α from a noise time series via
        linear regression on log(PSD) vs log(f), using Welch's averaged
        periodogram for low-variance spectral estimation.

        Args:
            x_t         : Noise time series.
            sample_rate : Sampling rate (Hz).

        Returns:
            (alpha_fit, r_squared): Fitted exponent and R² of the log-log fit.
        """
        from scipy.signal import welch
        N = len(x_t)
        # Welch segments: 8 averages for good variance reduction
        nperseg = max(256, N // 8)
        freqs, psd = welch(x_t, fs=sample_rate, nperseg=nperseg, scaling='density')

        # Fit over 1% to 40% of Nyquist (avoid DC and Nyquist edge effects)
        f_lo  = 0.01 * sample_rate / 2.0
        f_hi  = 0.40 * sample_rate / 2.0
        mask  = (freqs > f_lo) & (freqs < f_hi)

        log_f   = np.log10(freqs[mask])
        log_psd = np.log10(psd[mask])

        # Linear regression: log(PSD) = -α * log(f) + const
        coeffs = np.polyfit(log_f, log_psd, 1)
        alpha_fit = -coeffs[0]

        # R² calculation
        log_psd_fit = np.polyval(coeffs, log_f)
        ss_res = np.sum((log_psd - log_psd_fit)**2)
        ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return float(alpha_fit), float(r2)


# ─────────────────────────────────────────────────────────────────────────────
# Charge Offset Drift (Ornstein-Uhlenbeck)
# ─────────────────────────────────────────────────────────────────────────────

class ChargeOffsetDrift:
    """
    Generates slow charge offset drift using an Ornstein-Uhlenbeck (OU) process.

    The OU process models a mean-reverting random walk:
        d(δV) = -δV/τ * dt + σ * dW
    where dW = ξ√dt is a Wiener increment.

    Discretised Euler-Maruyama scheme:
        δV(t+dt) = δV(t) * (1 - dt/τ) + σ * sqrt(dt) * ξ(t)

    This produces a stationary process with:
        Mean     : ⟨δV⟩ = 0
        Variance : Var(δV) = σ²τ/2
        Autocorr : C(τ') = σ²τ/2 * exp(-|τ'|/τ)

    Physical interpretation: The charge trap ensemble slowly randomises the
    effective gate offset on timescales τ (seconds), causing slow drift of
    the charge stability diagram.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.cfg = config or DriftConfig()

    @property
    def stationary_variance(self) -> float:
        """Stationary variance of the OU process: σ²τ/2."""
        return (self.cfg.sigma**2 * self.cfg.tau) / 2.0

    @property
    def stationary_std(self) -> float:
        """Stationary standard deviation."""
        return np.sqrt(self.stationary_variance)

    def generate(self, n_samples: int, dt: float,
                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generates an OU process time series.

        Args:
            n_samples : Number of time steps.
            dt        : Time step size (s).
            seed      : Optional random seed.

        Returns:
            drift: Array of shape (n_samples,) in Volts.
        """
        rng    = np.random.default_rng(seed if seed is not None else self.cfg.seed)
        drift  = np.zeros(n_samples)
        drift[0] = self.cfg.initial

        decay  = 1.0 - dt / self.cfg.tau
        diffus = self.cfg.sigma * np.sqrt(dt)
        xi     = rng.standard_normal(n_samples - 1)

        for i in range(1, n_samples):
            drift[i] = decay * drift[i - 1] + diffus * xi[i - 1]

        logger.debug(f"ChargeOffsetDrift: τ={self.cfg.tau:.1f}s, "
                     f"σ_stat={self.stationary_std*1e6:.3f} µV")
        return drift


# ─────────────────────────────────────────────────────────────────────────────
# Composite Noise Generator
# ─────────────────────────────────────────────────────────────────────────────

class CompositeNoiseGenerator:
    """
    Combines all noise sources into a single gate voltage noise time series.

    Usage:
        gen = CompositeNoiseGenerator(
            johnson_cfg = JohnsonConfig(R=1e3, T=0.1),
            rtn_cfg     = RTNConfig(amplitude=50e-6, gamma_c=100, gamma_e=150),
            pink_cfg    = PinkNoiseConfig(amplitude=100e-6, alpha=1.0),
            drift_cfg   = DriftConfig(tau=10.0, sigma=1e-6),
        )
        noise = gen.generate(n_samples=8192, sample_rate=1e4)
    """

    def __init__(self,
                 johnson_cfg : Optional[JohnsonConfig]  = None,
                 rtn_cfg     : Optional[RTNConfig]      = None,
                 pink_cfg    : Optional[PinkNoiseConfig] = None,
                 drift_cfg   : Optional[DriftConfig]    = None):
        self.johnson = JohnsonNoise(johnson_cfg)
        self.rtn     = RTNoise(rtn_cfg)
        self.pink    = PinkNoise(pink_cfg)
        self.drift   = ChargeOffsetDrift(drift_cfg)

    def generate(self,
                 n_samples  : int,
                 sample_rate: float,
                 dt_drift   : Optional[float] = None,
                 seed       : int = 42,
                 include    : Tuple[bool, bool, bool, bool] = (True, True, True, True)
                 ) -> np.ndarray:
        """
        Generates a composite noise time series by superposing all sources.

        Args:
            n_samples   : Number of time steps.
            sample_rate : Sampling rate for Johnson, RTN, and Pink noise (Hz).
            dt_drift    : Time step for OU drift (s). Defaults to 1/sample_rate.
            seed        : Master random seed.
            include     : Tuple of (johnson, rtn, pink, drift) booleans.

        Returns:
            total_noise: Array of shape (n_samples,) in Volts.
        """
        rng   = np.random.default_rng(seed)
        total = np.zeros(n_samples)
        dt    = dt_drift or 1.0 / sample_rate

        if include[0]:
            total += self.johnson.generate(n_samples, sample_rate,
                                           seed=int(rng.integers(0, 2**32)))
        if include[1]:
            total += self.rtn.generate(n_samples, sample_rate,
                                       seed=int(rng.integers(0, 2**32)))
        if include[2]:
            pink_ts, _ = self.pink.generate(n_samples, sample_rate,
                                             seed=int(rng.integers(0, 2**32)))
            total += pink_ts
        if include[3]:
            total += self.drift.generate(n_samples, dt,
                                         seed=int(rng.integers(0, 2**32)))

        return total
