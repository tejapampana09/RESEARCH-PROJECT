"""
Constant Interaction Model for Multi-Dot Charge Stability
=========================================================

This module implements the Constant Interaction (CI) model, the standard physical
framework for describing the electrostatic energy and charge configurations of
coupled semiconductor quantum dot arrays.

Mathematical Model
------------------
The total electrostatic and quantum energy of a charge configuration
N = (N_1, N_2, ..., N_M)^T under gate voltages V_g = (V_{g1}, V_{g2}, ..., V_{gK})^T is:

    E(N, V_g) = 1/2 * N^T * U * N - e * N^T * alpha * V_g + \\sum_{i=1}^M \\sum_{n=1}^{N_i} E_{n, i}

where:
- U = e^2 * C^{-1} is the charging energy matrix.
  C is the capacitance matrix where:
    C_{ii} is the total capacitance of dot i,
    C_{ij} is the mutual capacitance between dot i and dot j (entered as a negative value).
- alpha = C_g * C^{-1} is the lever-arm matrix coupling gate voltages to dots.
- E_{n, i} is the single-particle confinement energy of the n-th electron in dot i.
- e is the elementary charge.

Stable Charge Configurations
----------------------------
At zero temperature, the system occupies the charge state N* that minimizes the energy:
    N*(V_g) = argmin_{N} E(N, V_g)

At finite temperature T, the charge state is thermally broadened. The probability of
a configuration N is given by the Gibbs distribution:
    P(N) = exp(-E(N, V_g) / (k_B * T)) / Z
    Z = \\sum_{N'} exp(-E(N', V_g) / (k_B * T))

The expectation value of the occupation on dot i is:
    <N_i> = \\sum_{N} N_i * P(N)

Readout / Charge Sensor
-----------------------
Experimental charge stability diagrams are measured using a charge sensor (e.g., an adjacent
quantum point contact or single electron transistor). The sensor conductance G is sensitive to
the electrostatic potential of the dots:
    G(V_g) = G_0 + \\sum_i w_i * <N_i> + Noise
where w_i represents the sensor coupling sensitivity to dot i. The derivative dG/dV_g shows
peaks at charge transitions.

References
----------
1. W. G. van der Wiel et al., "Electron transport through double quantum dots," Reviews of Modern
   Physics 75, 1 (2002).
2. C. Volk et al., "A charge sensor for a double silicon quantum dot," Applied Physics Letters
   98, 063109 (2011).
"""

import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional

# Boltzmann constant in eV/K
KB_EV = 8.617333262145e-5

class ConstantInteractionModel:
    """
    Implements the Constant Interaction model for a multi-quantum dot array.
    """
    def __init__(self, 
                 num_dots: int,
                 charging_matrix: Optional[np.ndarray] = None,
                 lever_arms: Optional[np.ndarray] = None,
                 capacitance_matrix: Optional[np.ndarray] = None,
                 gate_capacitance_matrix: Optional[np.ndarray] = None):
        """
        Initializes the model. Can specify either the charging and lever arm matrices directly
        (in eV), or provide the capacitance matrices (in Farads) to compute them.
        
        Args:
            num_dots: Number of quantum dots.
            charging_matrix: U matrix of shape (M, M) in eV.
            lever_arms: alpha matrix of shape (M, K) in eV/V.
            capacitance_matrix: C matrix of shape (M, M) in Farads.
            gate_capacitance_matrix: C_g matrix of shape (M, K) in Farads.
        """
        self.num_dots = num_dots
        self.e = 1.0  # eV/V
        
        if charging_matrix is not None:
            self.U = np.array(charging_matrix, dtype=float)
        elif capacitance_matrix is not None:
            # Compute U = e^2 * C^{-1}
            # Note: in physical formulas, e = 1.602e-19 C.
            # To get U in eV, U = e * C^{-1} (in Volts * e)
            C_inv = np.linalg.inv(capacitance_matrix)
            self.U = 1.602176634e-19 * C_inv  # converts to eV
        else:
            # Default double-dot charging energies: E_c1 = 4.0 meV, E_c2 = 4.0 meV, E_cm = 0.8 meV
            self.U = np.array([[4.0e-3, 0.8e-3], 
                               [0.8e-3, 4.0e-3]])
            
        if lever_arms is not None:
            self.alpha = np.array(lever_arms, dtype=float)
        elif gate_capacitance_matrix is not None and capacitance_matrix is not None:
            # alpha = C_g * C^{-1}
            C_inv = np.linalg.inv(capacitance_matrix)
            self.alpha = np.dot(gate_capacitance_matrix, C_inv)
        else:
            # Default double-dot lever arms: plunger 1 couples to dot 1, plunger 2 couples to dot 2
            # with 10% cross-talk
            self.alpha = np.array([[0.12, 0.012], 
                                   [0.012, 0.12]])

    def calculate_energy(self, 
                         N: np.ndarray, 
                         V_g: np.ndarray, 
                         single_particle_energies: Optional[List[List[float]]] = None) -> float:
        """
        Computes the total energy (eV) of a given charge configuration.
        
        Args:
            N: Charge configuration vector of shape (M,) containing integer electron occupations.
            V_g: Gate voltage vector of shape (K,) in Volts.
            single_particle_energies: Nested list where single_particle_energies[i][n] is the
                                     n-th energy level (eV) of dot i.
                                     If None, we assume a simple harmonic spectrum.
        """
        N = np.array(N, dtype=float)
        V_g = np.array(V_g, dtype=float)
        
        # Electrostatic term: 1/2 * N^T * U * N - e * N^T * alpha * V_g
        electrostatic = 0.5 * np.dot(N, np.dot(self.U, N)) - self.e * np.dot(N, np.dot(self.alpha, V_g))
        
        # Quantum confinement term: sum over dots and occupied states
        quantum_confinement = 0.0
        for i in range(self.num_dots):
            num_electrons = int(round(N[i]))
            if num_electrons > 0:
                if single_particle_energies is not None and len(single_particle_energies[i]) >= num_electrons:
                    quantum_confinement += sum(single_particle_energies[i][:num_electrons])
                else:
                    # Fallback: simple 2D harmonic spectrum E_n = (n + 1) * hbar_omega
                    # spacing of ~0.5 meV
                    spacing = 0.5e-3  # eV
                    quantum_confinement += sum((n + 1) * spacing for n in range(num_electrons))
                    
        return electrostatic + quantum_confinement

    def generate_charge_states(self, max_electrons: int = 4) -> List[Tuple[int, ...]]:
        """
        Generates all possible charge states N up to max_electrons on any dot.
        """
        ranges = [range(max_electrons + 1) for _ in range(self.num_dots)]
        return list(itertools.product(*ranges))

    def find_ground_state(self, 
                          V_g: np.ndarray, 
                          max_electrons: int = 4, 
                          single_particle_energies: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, float]:
        """
        Finds the ground state charge configuration N* at T = 0 K.
        
        Returns:
            N_star: The ground state charge vector (M,).
            min_energy: The energy of the ground state in eV.
        """
        states = self.generate_charge_states(max_electrons)
        min_energy = float('inf')
        N_star = np.zeros(self.num_dots, dtype=int)
        
        for state in states:
            N = np.array(state)
            E = self.calculate_energy(N, V_g, single_particle_energies)
            if E < min_energy:
                min_energy = E
                N_star = N
                
        return N_star, min_energy

    def get_thermal_occupations(self, 
                                V_g: np.ndarray, 
                                T_K: float = 0.1, 
                                max_electrons: int = 4, 
                                single_particle_energies: Optional[List[List[float]]] = None) -> np.ndarray:
        """
        Computes the thermal expectation value of electron occupation <N> on each dot at temperature T.
        
        Args:
            V_g: Plunger voltages.
            T_K: Temperature in Kelvin.
            max_electrons: Limit of grid check.
            
        Returns:
            1D array of shape (M,) representing average occupations <N_i>.
        """
        states = self.generate_charge_states(max_electrons)
        energies = np.zeros(len(states))
        
        for idx, state in enumerate(states):
            N = np.array(state)
            energies[idx] = self.calculate_energy(N, V_g, single_particle_energies)
            
        # Shift energies to avoid numerical overflow when calculating exp(-E / kBT)
        min_E = np.min(energies)
        shifted_energies = energies - min_E
        
        # Calculate Partition Function Z
        kBT = KB_EV * T_K
        
        # Avoid division by zero if T = 0
        if kBT < 1e-8:
            # Just return T=0 ground state
            N_star, _ = self.find_ground_state(V_g, max_electrons, single_particle_energies)
            return N_star.astype(float)
            
        weights = np.exp(-shifted_energies / kBT)
        Z = np.sum(weights)
        
        probabilities = weights / Z
        
        # Calculate expectations <N_i> = sum_k N_i^(k) * P(k)
        expectations = np.zeros(self.num_dots)
        for idx, state in enumerate(states):
            expectations += np.array(state) * probabilities[idx]
            
        return expectations

    def get_sensor_signal(self, 
                          V_g: np.ndarray, 
                          T_K: float = 0.1, 
                          max_electrons: int = 4, 
                          sensor_weights: Optional[np.ndarray] = None, 
                          single_particle_energies: Optional[List[List[float]]] = None) -> float:
        """
        Simulates the readout of a charge sensor.
        
        Args:
            sensor_weights: Coupling coefficients w_i of the sensor to dot i.
                            Defaults to different values for each dot (e.g. [1.0, 0.7, 0.4])
                            so that charge transitions can be visually distinguished.
        """
        if sensor_weights is None:
            # Default weights: distinguish dots by step size
            sensor_weights = np.array([1.0 - 0.25 * i for i in range(self.num_dots)])
            
        expectations = self.get_thermal_occupations(V_g, T_K, max_electrons, single_particle_energies)
        # Sensor response: we subtract a baseline, or just return the sum of weighted occupations
        return float(np.dot(sensor_weights, expectations))
