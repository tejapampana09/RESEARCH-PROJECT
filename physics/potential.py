"""
Potential Landscape Model for Silicon Quantum Dot Arrays
========================================================

This module implements the calculation of the 2D electrostatic potential landscape
V(x, y) experienced by electrons in a silicon-based quantum dot array.

Mathematical Model
------------------
The total potential energy V(x, y) (in eV) is the sum of:
1. Confinement potential from the quantum dot physical structures (e.g. gate-defined or self-assembled):
   V_conf(x, y) = - \\sum_{d} V_{depth, d} \\exp\\left( - \\frac{(x - x_d)^2 + (y - y_d)^2}{2 w_d^2} \\right)
   Near the center, this behaves harmonically with an effective spring constant:
   k_d = \\frac{V_{depth, d}}{w_d^2} = m^* \\omega_d^2

2. Electrostatic influence from control gates (plungers and barriers):
   V_gates(x, y) = -e \\sum_{k} \\alpha_k V_k \\exp\\left( - \\frac{(x - x_k)^2 + (y - y_k)^2}{2 \\sigma_k^2} \\right)
   where V_k is the voltage applied to gate k, \\alpha_k is the gate lever arm (coupling efficiency),
   and \\sigma_k is the spatial footprint of the gate's electric field.

Parameters for Silicon
----------------------
- Effective mass (transverse): m^* \\approx 0.19 * m_e (where m_e is the free electron mass).
- Dielectric constant of Silicon: \\epsilon_{Si} \\approx 11.7 * \\epsilon_0.

References
----------
1. F. A. Zwanenburg et al., "Silicon quantum electronics," Reviews of Modern Physics 85, 961 (2013).
2. C. Kloeffel and D. Loss, "Prospects for Spin-Based Quantum Computing in Silicon and Germanium,"
   Annual Review of Condensed Matter Physics 4, 51-81 (2013).
"""

import numpy as np
from typing import Dict, List, Tuple, Any

class PotentialLandscape:
    """
    Computes the 2D electrostatic potential energy landscape V(x, y) for a quantum dot array.
    All dimensions are in SI units (meters, seconds, electron-volts).
    """
    def __init__(self, 
                 grid_size: Tuple[int, int] = (100, 100), 
                 extent: Tuple[float, float] = (120e-9, 120e-9)):
        """
        Initializes the potential landscape grid.
        
        Args:
            grid_size: Number of grid points (Nx, Ny).
            extent: Spatial size of the simulation domain (Lx, Ly) in meters.
        """
        self.Nx, self.Ny = grid_size
        self.Lx, self.Ly = extent
        
        # Create grid coordinates (centered at 0, 0)
        self.x = np.linspace(-self.Lx / 2, self.Lx / 2, self.Nx)
        self.y = np.linspace(-self.Ly / 2, self.Ly / 2, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.dots: List[Dict[str, Any]] = []
        self.gates: List[Dict[str, Any]] = []
        
        # Physical constants
        self.e = 1.0  # We work in units of eV, so the electron charge magnitude is 1.0 (eV/V)
        self.m_e = 9.10938356e-31  # kg
        self.m_star = 0.19 * self.m_e  # Transverse effective mass in Si

    def add_dot(self, name: str, x0: float, y0: float, depth: float = 80e-3, radius: float = 12e-9):
        """
        Adds a structural quantum dot confinement well.
        
        Args:
            name: Identifier for the dot.
            x0: X coordinate of dot center (meters).
            y0: Y coordinate of dot center (meters).
            depth: Confinement potential depth (eV).
            radius: Radius of the dot (meters), corresponding to the standard deviation.
        """
        self.dots.append({
            'name': name,
            'x0': x0,
            'y0': y0,
            'depth': depth,
            'radius': radius
        })

    def add_gate(self, name: str, x0: float, y0: float, 
                 gate_type: str = "plunger", lever_arm: float = 0.12, sigma: float = 18e-9):
        """
        Adds a control gate electrode.
        
        Args:
            name: Identifier for the gate.
            x0: X coordinate of the gate center (meters).
            y0: Y coordinate of the gate center (meters).
            gate_type: "plunger" (for dot charge tuning) or "barrier" (for coupling control).
            lever_arm: Dimensionless coupling constant alpha (eV/V).
            sigma: Spatial spread of the gate's potential influence (meters).
        """
        self.gates.append({
            'name': name,
            'x0': x0,
            'y0': y0,
            'type': gate_type,
            'lever_arm': lever_arm,
            'sigma': sigma
        })

    def get_potential(self, gate_voltages: Dict[str, float]) -> np.ndarray:
        """
        Computes the total potential energy landscape V(x, y) in eV for the given gate voltages.
        
        Args:
            gate_voltages: Dictionary mapping gate names to applied voltages in Volts.
            
        Returns:
            2D numpy array of shape (Nx, Ny) representing V(x, y) in eV.
        """
        # Start with zero potential
        V = np.zeros((self.Nx, self.Ny))
        
        # 1. Add structural quantum dot confinement wells (negative energy lowers well for electrons)
        for dot in self.dots:
            r2 = (self.X - dot['x0'])**2 + (self.Y - dot['y0'])**2
            V += -dot['depth'] * np.exp(-r2 / (2 * dot['radius']**2))
            
        # 2. Add electrostatic potential shifts from gates
        for gate in self.gates:
            voltage = gate_voltages.get(gate['name'], 0.0)
            r2 = (self.X - gate['x0'])**2 + (self.Y - gate['y0'])**2
            # For electrons, a positive gate voltage creates an attractive potential (-e * V_gate)
            # Thus, potential energy energy shift is -e * alpha * V_g
            V_shift = -self.e * gate['lever_arm'] * voltage * np.exp(-r2 / (2 * gate['sigma']**2))
            V += V_shift
            
        return V

    def get_harmonic_oscillator_frequency(self, depth: float, radius: float) -> float:
        """
        Calculates the effective harmonic oscillator frequency omega (rad/s)
        at the bottom of a Gaussian well.
        
        The Gaussian well is V(r) = -V0 * exp(-r^2 / 2s^2)
        Taylor expansion: V(r) approx -V0 + V0 * r^2 / (2s^2)
        Matching with 1/2 * m* * w^2 * r^2:
        w = sqrt( V0 / (m* * s^2) )
        
        Args:
            depth: Confinement depth in eV.
            radius: Confinement radius (sigma) in meters.
        """
        # Convert depth from eV to Joules
        V0_J = depth * 1.602176634e-19
        omega = np.sqrt(V0_J / (self.m_star * radius**2))
        return omega
