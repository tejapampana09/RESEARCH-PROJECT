"""
RF Reflectometry and Dispersive Charge Sensing Simulator
=========================================================

This module simulates the dispersive readout of quantum dot arrays using RF reflectometry
coupled to a gate electrode. It models impedance shifts, quantum capacitance C_q,
complex reflection coefficients Gamma, and In-phase (I) / Quadrature (Q) demodulated signals.

Mathematical Model
------------------
1. Resonator Impedance:
   A parallel RLC resonator has impedance:
       Z_p(w) = 1 / ( 1/R_p + i*w*C_tot + 1/(i*w*L_p) )
   where:
       C_tot = C_p + C_q
       C_p is the parasitic capacitance (~1 pF).
       L_p is the resonator inductance (~100 nH).
       R_p = Q * w_0 * L_p is the resonator loss resistance.
       w_0 = 1 / sqrt(L_p * C_p) is the unperturbed resonance frequency.

2. Quantum Capacitance:
   C_q represents the capacitive loading due to coherent tunneling. It is proportional
   to the second derivative (curvature) of the ground state energy E_0:
       C_q = -e^2 * d^2(E_0) / dV_g^2

3. Reflection Coefficient:
   The complex reflection coefficient at the matching interface (Z_0 = 50 Ohm) is:
       Gamma(w) = ( Z_p(w) - Z_0 ) / ( Z_p(w) + Z_0 )
   We measure the phase shift Delta_theta relative to the off-resonance baseline Gamma_0.

4. Demodulated Signals:
   Homodyne mixing outputs:
       I = V_in * |Gamma| * cos(Delta_theta)
       Q = V_in * |Gamma| * sin(Delta_theta)

References
----------
1. J. I. Colless et al., "Dispersive readout of a silicon quantum dot mono-layer," Physical Review
   Letters 110, 046805 (2013).
2. A. C. Betz et al., "Dispersive sensing of charge carrier dynamics in silicon gate-defined
   quantum dots," Nano Letters 15, 4622-4627 (2015).
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional

class RFReflectometrySimulator:
    """
    Simulates LC resonator circuit profiles and RF reflectometry phase/IQ loops
    for dispersive charge sensing.
    """
    def __init__(self,
                 L_p: float = 120e-9,      # Inductance (H)
                 C_p: float = 1.0e-12,      # Parasitic capacitance (F)
                 Q: float = 80.0,           # Quality factor
                 Z_0: float = 50.0,         # Line impedance (Ohm)
                 V_in: float = 1e-3):       # RF carrier amplitude (V)
        """
        Args:
            L_p: Resonator inductance in Henrys.
            C_p: Parasitic capacitance in Farads.
            Q: Quality factor.
            Z_0: Transmission line impedance (typically 50 Ohm).
            V_in: Input RF voltage amplitude.
        """
        self.L_p = L_p
        self.C_p = C_p
        self.Q = Q
        self.Z_0 = Z_0
        self.V_in = V_in
        self.e = 1.602176634e-19  # Electron charge (C)
        
        # Calculate unperturbed resonance frequency w_0 (rad/s) and loss resistance R_p (Ohm)
        self.w_0 = 1.0 / np.sqrt(self.L_p * self.C_p)
        self.f_0 = self.w_0 / (2.0 * np.pi)
        self.R_p = self.Q * self.w_0 * self.L_p

    def compute_quantum_capacitance(self,
                                    E_minus: float,
                                    E_mid: float,
                                    E_plus: float,
                                    dV: float = 20e-6) -> float:
        """
        Calculates the quantum capacitance C_q from ground state energies
        using the central difference approximation of the second derivative.
        
        C_q = -e^2 * d^2(E_0) / dV_g^2
        
        Note: Since E_minus, E_mid, E_plus are in eV, we multiply by the
        electron charge factor e to convert eV to Joules, making:
        C_q = -e * d^2(E_eV)/dV^2 (F).
        
        Args:
            E_minus: Energy at V_g - dV (eV).
            E_mid: Energy at V_g (eV).
            E_plus: Energy at V_g + dV (eV).
            dV: Voltage perturbation step (V).
        """
        if dV <= 0.0:
            raise ValueError("Voltage perturbation step dV must be positive.")
            
        # Numerical second derivative in eV/V^2
        d2E_eV_dV2 = (E_plus - 2.0 * E_mid + E_minus) / (dV**2)
        
        # Unit derivation:
        #   C_q = -e^2 * d^2E_J/dV_g^2
        #   d^2E_J/dV^2 = e * d^2E_eV/dV^2   (since E_J = e * E_eV)
        #   => C_q = -e^2 * e * d^2E_eV/dV^2 ... but d^2E_eV/dV^2 has units eV/V^2 = J/(e*V^2)
        #   => C_q = -e^2 * (J/(e*V^2)) = -e * J/V^2 = -e * F  (Farads)
        #   => C_q [F] = -e [C] * d^2E_eV/dV^2 [eV/V^2]
        #   This is correct because 1 eV/V^2 = e J / V^2 = e * (C/F) and -e * (e/F) = -e^2/F
        #   In practice: C_q = -e * (d2E_eV/dV^2)
        C_q = -self.e * d2E_eV_dV2
        
        # Clip unphysical values from numerical noise at sharp corners
        max_Cq = 5.0e-12  # 5 pF cap limit
        return float(np.clip(C_q, -max_Cq, max_Cq))

    def get_resonator_impedance(self, w: float, C_q: float) -> complex:
        """
        Computes the complex impedance Z_p(w) of the resonator including quantum capacitance C_q.
        
        1/Z = 1/R_p + i*w*(C_p + C_q) + 1/(i*w*L_p)
        """
        C_tot = self.C_p + C_q
        # Avoid division by zero at w=0
        if w < 1e-3:
            return complex(0.0, 0.0)
            
        inv_Z = 1.0 / self.R_p + 1j * w * C_tot + 1.0 / (1j * w * self.L_p)
        return 1.0 / inv_Z

    def get_reflection(self, f: float, C_q: float = 0.0) -> Tuple[complex, float, float]:
        """
        Computes the complex reflection coefficient Gamma at frequency f.
        
        Returns:
            Gamma: Complex reflection coefficient.
            amplitude: Absolute magnitude of Gamma.
            phase: Phase angle in radians.
        """
        w = 2.0 * np.pi * f
        Z_p = self.get_resonator_impedance(w, C_q)
        
        # Reflection Gamma = (Z_p - Z_0) / (Z_p + Z_0)
        Gamma = (Z_p - self.Z_0) / (Z_p + self.Z_0)
        
        return Gamma, float(np.abs(Gamma)), float(np.angle(Gamma))

    def get_demodulated_iq(self,
                           Gamma: complex,
                           Gamma_ref: Optional[complex] = None) -> Tuple[float, float, float]:
        """
        Demodulates the complex reflection coefficient into In-phase (I) and
        Quadrature (Q) components, and returns the phase shift Delta_theta.
        
        Args:
            Gamma: The complex reflection coefficient under test.
            Gamma_ref: Reference complex reflection (off-resonance or zero C_q).
                       If None, reference phase is assumed to be 0.
        """
        phase_test = np.angle(Gamma)
        phase_ref = np.angle(Gamma_ref) if Gamma_ref is not None else 0.0
        
        # Phase shift
        Delta_theta = float(np.unwrap([phase_ref, phase_test])[1] - phase_ref)
        
        # Demodulate signals
        amp = np.abs(Gamma)
        I = float(self.V_in * amp * np.cos(Delta_theta))
        Q = float(self.V_in * amp * np.sin(Delta_theta))
        
        return I, Q, Delta_theta
