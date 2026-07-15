"""
Digital Twin Engine and Real-Time State Synchronizer
=====================================================

This module implements the DigitalTwinEngine class, which serves as the core real-time
synchronizer. It couples the physical state parameters, controls the simulation loop,
and generates realistic experimental noise, including 1/f^alpha spectral noise
and thermal fluctuations.

1/f^alpha Noise Generation
--------------------------
Charge noise in silicon quantum dots is dominated by two-level fluctuators (TLFs)
in the surrounding oxide, yielding a spectral density:
    S(f) = A / f^alpha
where alpha approx 1.0. To simulate this in the time domain:
1. Generate random phase and Gaussian amplitude white noise in the frequency domain.
2. Scale the amplitude of each frequency component f by 1 / f^(alpha/2).
3. Perform the Inverse Fast Fourier Transform (IFFT).
4. Extract the real part and scale to the desired noise amplitude.

References
----------
1. E. Paladino et al., "1/f noise: Implications for solid-state quantum information," Reviews of
   Modern Physics 86, 361 (2014).
2. J. M. Boter et al., "Spatial correlation of charge noise in silicon-based quantum dots,"
   Physical Review B 101, 235433 (2020).
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
from simulator.device import SiliconQDArray

class NoiseGenerator:
    """
    Generates 1/f^alpha noise and white noise for realistic experimental simulations.
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    def generate_one_over_f(self, 
                            length: int, 
                            dt: float, 
                            amplitude: float = 1.0, 
                            alpha: float = 1.0) -> np.ndarray:
        """
        Generates a time series of 1/f^alpha noise.
        
        Args:
            length: Number of time points.
            dt: Time step in seconds.
            amplitude: Standard deviation scaling factor.
            alpha: Exponent (alpha=1 for pink noise, alpha=2 for brown noise).
        """
        # We need an even number of points for simple FFT
        N = length
        if N % 2 != 0:
            N += 1
            
        # Frequency grid
        freqs = np.fft.rfftfreq(N, d=dt)
        # Avoid division by zero at f=0
        freqs[0] = freqs[1]
        
        # Scale factor 1/f^(alpha/2)
        scale = 1.0 / (freqs ** (alpha / 2.0))
        
        # Generate random complex numbers (Gaussian amplitudes, uniform phases)
        real_noise = np.random.normal(0, 1, len(freqs))
        imag_noise = np.random.normal(0, 1, len(freqs))
        complex_noise = real_noise + 1j * imag_noise
        
        # Shape the spectrum
        shaped_noise = complex_noise * scale
        # DC component should be real and zero (no drift)
        shaped_noise[0] = 0.0
        
        # Inverse Fourier transform
        time_series = np.fft.irfft(shaped_noise, n=N)
        
        # Normalize to standard deviation = 1 and scale by amplitude
        std = np.std(time_series)
        if std > 0:
            time_series = (time_series / std) * amplitude
            
        return time_series[:length]

class DigitalTwinEngine:
    """
    Synchronizes the simulator state and handles physical parameter changes,
    adding real-time noise logs and mimicking experimental feedback loops.
    """
    def __init__(self, num_dots: int = 2):
        self.device = SiliconQDArray(num_dots=num_dots)
        self.noise_generator = NoiseGenerator()
        
        # Twin state parameters
        self.temperature_K: float = 0.1  # 100 mK
        self.noise_amplitude: float = 8.0e-5  # Volts (gate-referred noise)
        self.noise_exponent: float = 1.0  # 1/f noise
        self.dt: float = 0.01  # Time step of logger (seconds)
        
        # History log for dashboard charts
        self.time_history: List[float] = []
        self.noise_stream_history: List[float] = []
        self.sensor_history: List[float] = []
        
        self.start_time = time.time()
        self.steps_counter = 0

    def update_twin_parameters(self, params: Dict[str, Any]):
        """
        Updates the physical state of the digital twin.
        
        Args:
            params: Dictionary containing parameters to update, e.g.:
                    {"temperature": 0.05, "noise_amplitude": 1e-4, "gate_voltages": {...}}
        """
        if "temperature" in params:
            self.temperature_K = float(params["temperature"])
            
        if "noise_amplitude" in params:
            self.noise_amplitude = float(params["noise_amplitude"])
            
        if "noise_exponent" in params:
            self.noise_exponent = float(params["noise_exponent"])
            
        if "gate_voltages" in params:
            self.device.set_voltages(params["gate_voltages"])

    def get_synchronized_state(self) -> Dict[str, Any]:
        """
        Runs one step of the real-time simulation loop.
        Applies current gate-referred noise to the voltages and computes
        the updated charge occupations and sensor readings.
        """
        self.steps_counter += 1
        t_now = self.steps_counter * self.dt
        
        # 1. Generate local 1/f gate-referred voltage fluctuation
        # We model this as a noise voltage added to each plunger gate
        voltage_noise_vals = {}
        for gate_name in self.device.gate_voltages.keys():
            if gate_name.startswith("P"):  # Plunger noise is dominant
                # Generate a single step of pink noise by querying the generator
                # (We generate a chunk and take the last element for efficiency)
                noise_val = self.noise_generator.generate_one_over_f(
                    length=16, 
                    dt=self.dt, 
                    amplitude=self.noise_amplitude, 
                    alpha=self.noise_exponent
                )[-1]
                voltage_noise_vals[gate_name] = noise_val
                
        # 2. Compute temporary noisy gate voltages
        noisy_voltages = {}
        for gate_name, val in self.device.gate_voltages.items():
            noise = voltage_noise_vals.get(gate_name, 0.0)
            noisy_voltages[gate_name] = val + noise
            
        # 3. Temporarily apply noisy voltages to device to calculate charge states
        orig_voltages = self.device.gate_voltages.copy()
        self.device.set_voltages(noisy_voltages)
        
        charge_state = self.device.get_charge_state(T_K=self.temperature_K)
        # Sensor read, with added detector white noise (e.g. amplification noise)
        sensor_val = self.device.get_sensor_reading(T_K=self.temperature_K, noise_std=1e-3)
        
        # Restore nominal control voltages
        self.device.set_voltages(orig_voltages)
        
        # 4. Log history (keep last 100 points for real-time visualization)
        self.time_history.append(t_now)
        # Average noise over plungers for charting
        avg_noise = np.mean(list(voltage_noise_vals.values())) if voltage_noise_vals else 0.0
        self.noise_stream_history.append(avg_noise)
        self.sensor_history.append(sensor_val)
        
        if len(self.time_history) > 100:
            self.time_history.pop(0)
            self.noise_stream_history.pop(0)
            self.sensor_history.pop(0)
            
        return {
            "time": t_now,
            "gate_voltages": self.device.gate_voltages,
            "noisy_voltages": noisy_voltages,
            "charge_occupations": charge_state.tolist(),
            "sensor_reading": sensor_val,
            "noise_voltage": avg_noise,
            "history": {
                "time": self.time_history,
                "noise": self.noise_stream_history,
                "sensor": self.sensor_history
            }
        }
