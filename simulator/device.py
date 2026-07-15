"""
Silicon Quantum Dot Array Device Simulator
==========================================

This module implements the SiliconQDArray class, which integrates electrostatics,
quantum mechanics, and charging models to simulate a complete physical device
capable of modeling single, double, triple, or NxN quantum dot systems.

It exposes functions to:
1. Update gate voltages and compute potential landscapes.
2. Solve the 2D Schrödinger equation for the active electrostatic potential.
3. Compute stable charge configurations and sensor signals using the Constant Interaction model.
4. Generate 2D charge stability scans (honeycomb diagrams).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from physics.potential import PotentialLandscape
from physics.constant_interaction import ConstantInteractionModel
from physics.tunnel_coupling import TunnelCouplingModel
from quantum.schrodinger import SchrodingerSolver2D

class SiliconQDArray:
    """
    Represents a silicon quantum dot array device.
    Integrates the electrostatic potential, Schrödinger solver, and charging model.
    """
    def __init__(self, 
                 num_dots: int = 2,
                 grid_size: Tuple[int, int] = (80, 80),
                 extent: Tuple[float, float] = (120e-9, 120e-9)):
        """
        Initializes the quantum dot array device.
        
        Args:
            num_dots: Number of dots (e.g., 1 for single, 2 for double, 3 for triple).
            grid_size: Spatial grid size (Nx, Ny) for potential and wavefunction grids.
            extent: Spatial dimensions (Lx, Ly) in meters of the device.
        """
        self.num_dots = num_dots
        self.grid_size = grid_size
        self.extent = extent
        
        # Instantiate subcomponents
        self.potential_landscape = PotentialLandscape(grid_size=grid_size, extent=extent)
        self.schrodinger_solver = SchrodingerSolver2D()
        self.tunnel_model = TunnelCouplingModel()
        
        # Setup default configurations based on number of dots
        self.gate_voltages: Dict[str, float] = {}
        self._configure_device()

    def _configure_device(self):
        """
        Configures dots and gate layouts.
        For a double quantum dot (DQD), we place dots at (-25nm, 0) and (25nm, 0),
        with plunger gates P1, P2 and barrier gate B1.
        For a triple quantum dot (TQD), we place dots at (-35nm, 0), (0, 0), (35nm, 0),
        with plunger gates P1, P2, P3 and barrier gates B1, B2.
        """
        if self.num_dots == 1:
            # Single Quantum Dot
            self.potential_landscape.add_dot("dot1", x0=0.0, y0=0.0, depth=80e-3, radius=12e-9)
            self.potential_landscape.add_gate("P1", x0=0.0, y0=-30e-9, gate_type="plunger", lever_arm=0.15, sigma=18e-9)
            self.gate_voltages = {"P1": 0.0}
            
            # Charging Model
            # self-charging energy = 5 meV
            U = np.array([[5.0e-3]])
            alpha = np.array([[0.15]])
            self.ci_model = ConstantInteractionModel(num_dots=1, charging_matrix=U, lever_arms=alpha)
            self.sensor_weights = np.array([1.0])
            
        elif self.num_dots == 2:
            # Double Quantum Dot
            dot_spacing = 25e-9  # meters
            self.potential_landscape.add_dot("dot1", x0=-dot_spacing, y0=0.0, depth=80e-3, radius=12e-9)
            self.potential_landscape.add_dot("dot2", x0=dot_spacing, y0=0.0, depth=80e-3, radius=12e-9)
            
            # Plunger Gates
            self.potential_landscape.add_gate("P1", x0=-dot_spacing, y0=-35e-9, gate_type="plunger", lever_arm=0.12, sigma=16e-9)
            self.potential_landscape.add_gate("P2", x0=dot_spacing, y0=-35e-9, gate_type="plunger", lever_arm=0.12, sigma=16e-9)
            
            # Barrier Gate (between dots)
            self.potential_landscape.add_gate("B1", x0=0.0, y0=0.0, gate_type="barrier", lever_arm=0.08, sigma=12e-9)
            
            self.gate_voltages = {"P1": 0.0, "P2": 0.0, "B1": 0.2}
            
            # Charging Model
            # E_c1 = 4.0 meV, E_c2 = 4.0 meV, E_cm = 0.8 meV (mutual)
            U = np.array([[4.0e-3, 0.8e-3],
                          [0.8e-3, 4.0e-3]])
            
            # alpha: lever arm matrix (rows=dots, cols=gates [P1, P2])
            alpha = np.array([[0.12, 0.012],   # dot 1 coupling to P1, P2
                              [0.012, 0.12]])  # dot 2 coupling to P1, P2
            self.ci_model = ConstantInteractionModel(num_dots=2, charging_matrix=U, lever_arms=alpha)
            
            # Sensor weights: distinguish L and R dot transitions
            self.sensor_weights = np.array([1.0, 0.65])
            
        elif self.num_dots == 3:
            # Triple Quantum Dot
            dot_spacing = 35e-9
            self.potential_landscape.add_dot("dot1", x0=-dot_spacing, y0=0.0, depth=80e-3, radius=10e-9)
            self.potential_landscape.add_dot("dot2", x0=0.0, y0=0.0, depth=80e-3, radius=10e-9)
            self.potential_landscape.add_dot("dot3", x0=dot_spacing, y0=0.0, depth=80e-3, radius=10e-9)
            
            # Plunger Gates
            self.potential_landscape.add_gate("P1", x0=-dot_spacing, y0=-35e-9, gate_type="plunger", lever_arm=0.11, sigma=15e-9)
            self.potential_landscape.add_gate("P2", x0=0.0, y0=-35e-9, gate_type="plunger", lever_arm=0.11, sigma=15e-9)
            self.potential_landscape.add_gate("P3", x0=dot_spacing, y0=-35e-9, gate_type="plunger", lever_arm=0.11, sigma=15e-9)
            
            # Barrier Gates
            self.potential_landscape.add_gate("B1", x0=-dot_spacing/2.0, y0=0.0, gate_type="barrier", lever_arm=0.07, sigma=10e-9)
            self.potential_landscape.add_gate("B2", x0=dot_spacing/2.0, y0=0.0, gate_type="barrier", lever_arm=0.07, sigma=10e-9)
            
            self.gate_voltages = {"P1": 0.0, "P2": 0.0, "P3": 0.0, "B1": 0.2, "B2": 0.2}
            
            # Charging Model
            # Self charging energy: 3.5 meV. Neighboring coupling: 0.7 meV. Next-neighbor: 0.15 meV.
            U = np.array([[3.5e-3, 0.7e-3, 0.15e-3],
                          [0.7e-3, 3.5e-3, 0.7e-3],
                          [0.15e-3, 0.7e-3, 3.5e-3]])
            
            alpha = np.array([[0.11, 0.015, 0.002],  # dot 1 to plungers P1, P2, P3
                              [0.015, 0.11, 0.015],  # dot 2 to plungers
                              [0.002, 0.015, 0.11]])  # dot 3 to plungers
            self.ci_model = ConstantInteractionModel(num_dots=3, charging_matrix=U, lever_arms=alpha)
            
            self.sensor_weights = np.array([1.0, 0.7, 0.4])
            
        else:
            # General NxN or larger linear array layout (handled dynamically as linear array)
            # Default to linear array of M dots
            M = self.num_dots
            spacing = 20e-9
            start_x = -spacing * (M - 1) / 2.0
            
            for i in range(M):
                dot_x = start_x + i * spacing
                self.potential_landscape.add_dot(f"dot{i+1}", x0=dot_x, y0=0.0, depth=80e-3, radius=8e-9)
                self.potential_landscape.add_gate(f"P{i+1}", x0=dot_x, y0=-25e-9, gate_type="plunger", lever_arm=0.12, sigma=12e-9)
                self.gate_voltages[f"P{i+1}"] = 0.0
                if i < M - 1:
                    self.potential_landscape.add_gate(f"B{i+1}", x0=dot_x + spacing/2.0, y0=0.0, gate_type="barrier", lever_arm=0.08, sigma=8e-9)
                    self.gate_voltages[f"B{i+1}"] = 0.2
                    
            # Build simple matrices
            U = np.eye(M) * 4.0e-3
            for i in range(M-1):
                U[i, i+1] = 0.7e-3
                U[i+1, i] = 0.7e-3
            alpha = np.eye(M) * 0.12
            self.ci_model = ConstantInteractionModel(num_dots=M, charging_matrix=U, lever_arms=alpha)
            self.sensor_weights = np.array([1.0 - 0.15*i for i in range(M)])

    def set_voltages(self, voltages: Dict[str, float]):
        """
        Updates the gate voltage configurations.
        
        Args:
            voltages: Dict of {gate_name: voltage_value}
        """
        for k, v in voltages.items():
            if k in self.gate_voltages:
                self.gate_voltages[k] = v

    def get_potential_map(self) -> np.ndarray:
        """
        Computes the current 2D potential energy map in eV.
        """
        return self.potential_landscape.get_potential(self.gate_voltages)

    def solve_quantum_states(self, num_states: int = 5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Runs the Schrödinger solver on the current potential landscape.
        """
        V = self.get_potential_map()
        return self.schrodinger_solver.solve(V, self.potential_landscape.x, self.potential_landscape.y, num_states)

    def get_charge_state(self, T_K: float = 0.1, max_electrons: int = 4) -> np.ndarray:
        """
        Computes the thermal average charge occupation on each dot for current voltages.
        """
        # Collect only plunger voltages for the CI model.
        # Plunger names are "P1", "P2", etc.
        V_g = np.array([self.gate_voltages.get(f"P{i+1}", 0.0) for i in range(self.num_dots)])
        return self.ci_model.get_thermal_occupations(V_g, T_K=T_K, max_electrons=max_electrons)

    def get_sensor_reading(self, T_K: float = 0.1, noise_std: float = 0.0) -> float:
        """
        Simulates the readout sensor conductance.
        """
        V_g = np.array([self.gate_voltages.get(f"P{i+1}", 0.0) for i in range(self.num_dots)])
        signal = self.ci_model.get_sensor_signal(V_g, T_K=T_K, sensor_weights=self.sensor_weights)
        if noise_std > 0.0:
            signal += np.random.normal(0, noise_std)
        return signal

    def get_stability_diagram(self, 
                              v1_range: np.ndarray, 
                              v2_range: np.ndarray, 
                              gate1: str = "P1", 
                              gate2: str = "P2", 
                              T_K: float = 0.1, 
                              max_electrons: int = 3) -> Dict[str, np.ndarray]:
        """
        Generates the 2D charge stability diagram by scanning gate1 and gate2 voltages.
        
        Returns:
            A dictionary containing:
            - "sensor": 2D grid of sensor reading.
            - "charge1": 2D grid of dot 1 average occupation.
            - "charge2": 2D grid of dot 2 average occupation.
            - "derivative": 2D grid of the combined derivative (showing transition lines).
            - "V1", "V2": meshgrids of the scanned voltages.
        """
        Nv1 = len(v1_range)
        Nv2 = len(v2_range)
        
        sensor = np.zeros((Nv1, Nv2))
        charge1 = np.zeros((Nv1, Nv2))
        charge2 = np.zeros((Nv1, Nv2))
        
        # Save current voltages
        orig_voltages = self.gate_voltages.copy()
        
        # Determine gate mapping for charging model
        # The CI model input is plunger voltages. We map gate1 and gate2 indices.
        # Plunger names are "P1" (index 0), "P2" (index 1), "P3" (index 2), etc.
        idx1 = int(gate1[1:]) - 1 if (gate1.startswith("P") and gate1[1:].isdigit()) else 0
        idx2 = int(gate2[1:]) - 1 if (gate2.startswith("P") and gate2[1:].isdigit()) else 1
        
        # Standard scan
        for i, v1 in enumerate(v1_range):
            for j, v2 in enumerate(v2_range):
                # Update voltages
                self.gate_voltages[gate1] = v1
                self.gate_voltages[gate2] = v2
                
                V_g = np.array([self.gate_voltages.get(f"P{k+1}", 0.0) for k in range(self.num_dots)])
                
                occupations = self.ci_model.get_thermal_occupations(V_g, T_K=T_K, max_electrons=max_electrons)
                sensor_val = self.ci_model.get_sensor_signal(V_g, T_K=T_K, sensor_weights=self.sensor_weights)
                
                sensor[i, j] = sensor_val
                charge1[i, j] = occupations[idx1] if idx1 < len(occupations) else 0.0
                charge2[i, j] = occupations[idx2] if idx2 < len(occupations) else 0.0
                
        # Restore voltages
        self.gate_voltages = orig_voltages
        
        # Calculate gradients (numerical derivatives) to show the honeycomb boundaries
        # dS/dV1 and dS/dV2
        grad_v1, grad_v2 = np.gradient(sensor, v1_range, v2_range)
        # Combine the gradients using L1 norm or L2 norm for clear visualization of transitions
        derivative = np.sqrt(grad_v1**2 + grad_v2**2)
        
        # Meshgrids for outputs
        V1_mesh, V2_mesh = np.meshgrid(v1_range, v2_range, indexing='ij')
        
        return {
            "sensor": sensor,
            "charge1": charge1,
            "charge2": charge2,
            "derivative": derivative,
            "V1": V1_mesh,
            "V2": V2_mesh
        }
