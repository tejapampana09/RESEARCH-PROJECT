"""
Tunnel Coupling Model for Silicon Spin Qubits
==============================================

This module calculates the tunnel coupling t between adjacent quantum dots.
Tunnel coupling is a key parameter that controls the exchange interaction
between spin qubits.

Mathematical Models
-------------------
1. Wavefunction Overlap / Energy Splitting:
   For a symmetric double quantum dot at zero detuning (charge degeneracy point (1,0) <-> (0,1)),
   the tunnel coupling t is exactly half of the energy splitting between the ground state
   (symmetric bonding orbital) and the first excited state (antisymmetric anti-bonding orbital):

       t = (E_1 - E_0) / 2

2. Phenomenological Barrier Gate Dependency:
   The barrier gate voltage V_b modulates the height and width of the potential barrier.
   According to the WKB approximation, tunnel coupling depends exponentially on the barrier height V_0:
   
       t(V_b) = t_0 * exp( - beta * sqrt(V_{barrier_height} - V_b) )
       
   For simulation, this is commonly linearized in the exponent near the operating point:
   
       t(V_b) = t_max * exp( gamma * (V_b - V_{max}) )
       
   where:
   - V_b is the barrier gate voltage (V).
   - t_max is the maximum tunnel coupling at V_b = V_max (eV).
   - gamma is the sensitivity parameter (1/V), representing the gate lever arm on the barrier.

References
----------
1. L. R. Schreiber et al., "Coupling of single-electron spin qubits in silicon," Nature Communications
   2, 556 (2011).
2. F. R. Braakman et al., "Coherent transfer of a single electron spin through a triple quantum dot,"
   Nature Nanotechnology 8, 432-437 (2013).
"""

import numpy as np
from typing import Tuple, Dict

class TunnelCouplingModel:
    """
    Models and estimates tunnel coupling between quantum dots using numerical
    and phenomenological approaches.
    """
    def __init__(self, t_max: float = 120e-6, V_max: float = 0.5, gamma: float = 8.0):
        """
        Args:
            t_max: Maximum tunnel coupling in eV (default 120 ueV, which is approx 29 GHz).
            V_max: Barrier voltage at which t_max is achieved (Volts).
            gamma: Sensitivity parameter in V^-1 governing the exponential scaling.
        """
        self.t_max = t_max
        self.V_max = V_max
        self.gamma = gamma

    def get_coupling_phenom(self, V_barrier: float) -> float:
        """
        Computes the tunnel coupling t (eV) for a given barrier gate voltage (V)
        using the phenomenological exponential model.
        
        t(V_barrier) = t_max * exp( gamma * (V_barrier - V_max) )
        """
        # Ensure we cap the tunnel coupling to prevent unphysical values at large voltages
        exponent = self.gamma * (V_barrier - self.V_max)
        # Cap the exponent at 2.0 to prevent numerical explosion
        exponent = np.clip(exponent, -20.0, 2.0)
        return self.t_max * np.exp(exponent)

    def get_splitting_coupling(self, eigenvalues_eV: np.ndarray) -> float:
        """
        Calculates the tunnel coupling from the energy eigenvalues of a double dot
        at the charge degeneracy point (zero detuning).
        
        t = (E_1 - E_0) / 2
        
        Args:
            eigenvalues_eV: Array of eigenvalues in eV, sorted in ascending order.
            
        Returns:
            Tunnel coupling t in eV.
        """
        if len(eigenvalues_eV) < 2:
            raise ValueError("Need at least 2 eigenvalues to calculate splitting.")
        return float((eigenvalues_eV[1] - eigenvalues_eV[0]) / 2.0)

    def wkb_barrier_transmission(self, 
                                 barrier_width: float, 
                                 barrier_height_eV: float, 
                                 electron_energy_eV: float, 
                                 m_star: float = 0.19 * 9.109e-31) -> float:
        """
        Calculates the WKB transmission probability T through a rectangular potential barrier.
        
        T = exp(-2 * kappa * w)
        where:
        kappa = sqrt( 2 * m* * (V_0 - E) ) / hbar
        w is the barrier width.
        
        Args:
            barrier_width: Width of the barrier (meters).
            barrier_height_eV: Height of the barrier (eV).
            electron_energy_eV: Energy of the incident electron (eV).
            m_star: Effective mass of the electron (kg).
        """
        hbar = 1.054571817e-34  # J*s
        e = 1.602176634e-19     # C
        
        # If the electron energy is higher than the barrier, WKB transmission is 1
        if electron_energy_eV >= barrier_height_eV:
            return 1.0
            
        dE_J = (barrier_height_eV - electron_energy_eV) * e
        kappa = np.sqrt(2.0 * m_star * dE_J) / hbar
        
        transmission = np.exp(-2.0 * kappa * barrier_width)
        return float(transmission)
