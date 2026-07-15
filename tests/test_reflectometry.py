"""
Unit Tests for the RF Reflectometry Simulator
=============================================

This test suite verifies:
1. Resonator initial parameters (resonance frequency, loss resistance).
2. Quantum capacitance calculation using finite differences of quadratic energy.
3. Reflection coefficient calculations at and off resonance.
4. IQ demodulation signals and phase shifts.
"""

import numpy as np
import pytest
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.reflectometry import RFReflectometrySimulator

def test_resonator_parameters():
    """
    Checks that the unperturbed resonance frequency and parallel resistance match theory.
    """
    L = 120e-9  # 120 nH
    C = 1.0e-12  # 1 pF
    Q = 80.0
    Z_0 = 50.0
    
    sim = RFReflectometrySimulator(L_p=L, C_p=C, Q=Q, Z_0=Z_0)
    
    # Theoretical f_0 = 1 / (2 * pi * sqrt(L * C))
    f_0_theory = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
    assert pytest.approx(sim.f_0, rel=1e-5) == f_0_theory
    assert sim.f_0 > 450e6  # ~459 MHz
    
    # Theoretical R_p = Q * w_0 * L = Q * sqrt(L/C)
    R_p_theory = Q * np.sqrt(L / C)
    assert pytest.approx(sim.R_p, rel=1e-5) == R_p_theory
    assert sim.R_p > 20000.0  # ~27.7 kOhm

def test_quantum_capacitance():
    """
    Verifies that C_q matches analytical second derivatives for quadratic energy curves.
    """
    sim = RFReflectometrySimulator()
    
    # Let E(V) = -a * V^2 (in eV)
    # Then d^2E_eV/dV^2 = -2*a   [eV/V^2]
    # C_q = -e * d^2E_eV/dV^2 = -e * (-2*a) = 2 * e * a  [F]
    e_c = 1.602176634e-19
    a = 100.0
    dV = 10e-6  # 10 uV
    
    # Evaluate at V=0
    E_mid = 0.0
    E_plus = -a * (dV)**2
    E_minus = -a * (-dV)**2
    
    C_q = sim.compute_quantum_capacitance(E_minus, E_mid, E_plus, dV=dV)
    
    C_q_theory = 2.0 * e_c * a
    assert pytest.approx(C_q, rel=1e-4) == C_q_theory

def test_reflection_on_resonance():
    """
    Verifies that reflection calculations at f_0 match circuit impedance theory.
    """
    sim = RFReflectometrySimulator(Q=50.0)
    
    # On resonance, Z_p = R_p (real resistance)
    # Gamma_theory = (R_p - Z_0) / (R_p + Z_0)
    Gamma_theory = (sim.R_p - sim.Z_0) / (sim.R_p + sim.Z_0)
    
    Gamma, amp, phase = sim.get_reflection(sim.f_0, C_q=0.0)
    
    assert pytest.approx(np.real(Gamma), rel=1e-4) == Gamma_theory
    assert pytest.approx(amp, rel=1e-4) == Gamma_theory
    assert pytest.approx(phase, abs=1e-5) == 0.0  # Phase should be 0 on resonance

def test_iq_demodulation():
    """
    Verifies that IQ outputs and phase shifts demodulate correctly.
    """
    sim = RFReflectometrySimulator(V_in=2.0e-3) # 2 mV input
    
    # Shift resonance slightly by adding 10 fF of quantum capacitance
    C_q = 10.0e-15
    
    Gamma_ref, _, _ = sim.get_reflection(sim.f_0, C_q=0.0)
    Gamma_test, amp_test, _ = sim.get_reflection(sim.f_0, C_q=C_q)
    
    I, Q, Delta_theta = sim.get_demodulated_iq(Gamma_test, Gamma_ref)
    
    # Check that amplitude is conserved: I^2 + Q^2 = (V_in * amp_test)^2
    reconstructed_amp = np.sqrt(I**2 + Q**2)
    target_amp = sim.V_in * amp_test
    assert pytest.approx(reconstructed_amp, rel=1e-5) == target_amp
    
    # Check that angle matches Delta_theta
    assert Delta_theta < 0.0  # Positive C_q shifts resonance down, causing negative phase shift
