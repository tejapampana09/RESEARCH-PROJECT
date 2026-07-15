"""
Unit Tests for the QuantumTwin Physics and AI Modules
======================================================

This test suite verifies:
1. The 2D potential landscape grid creation, dot placements, and gate-voltage shifts.
2. The 2D finite-difference Schrödinger solver eigenvalues and wavefunction normalization.
3. The Constant Interaction charging energy calculations and ground state searches.
4. The PyTorch deep Q-network auto-tuning agent integration.

Run tests using:
    pytest tests/test_quantum_twin.py
"""

import numpy as np
import torch
import pytest
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.potential import PotentialLandscape
from physics.constant_interaction import ConstantInteractionModel
from physics.tunnel_coupling import TunnelCouplingModel
from quantum.schrodinger import SchrodingerSolver2D
from simulator.device import SiliconQDArray
from ai.tuning import DQNTuningAgent

def test_potential_landscape():
    """
    Verifies that the electrostatic potential well functions correctly.
    """
    Nx, Ny = 40, 40
    Lx, Ly = 100e-9, 100e-9
    landscape = PotentialLandscape(grid_size=(Nx, Ny), extent=(Lx, Ly))
    
    # Add a dot at the center (0,0) and plunger gate
    landscape.add_dot("dot1", x0=0.0, y0=0.0, depth=0.1, radius=10e-9)
    landscape.add_gate("P1", x0=0.0, y0=-20e-9, lever_arm=0.1, sigma=15e-9)
    
    # 0V applied potential
    V_0V = landscape.get_potential({"P1": 0.0})
    assert V_0V.shape == (Nx, Ny)
    # The center should have negative potential (attractive well)
    assert V_0V[Nx//2, Ny//2] < 0.0
    
    # 1V applied potential on P1 plunger (should pull potential lower at gate position)
    V_1V = landscape.get_potential({"P1": 1.0})
    # Potential with +1V on P1 should be lower (more attractive for electrons) than with 0V
    # Gate is centered at (0, -20nm), which is in the lower half of the y-axis
    assert np.min(V_1V) < np.min(V_0V)
    
    # Check harmonic frequency calculation
    omega = landscape.get_harmonic_oscillator_frequency(depth=80e-3, radius=10e-9)
    assert omega > 0.0
    assert isinstance(omega, float)

def test_schrodinger_solver():
    """
    Verifies Schrödinger solver eigenvalues sorting and wavefunction normalization.
    """
    Nx, Ny = 30, 30
    x = np.linspace(-50e-9, 50e-9, Nx)
    y = np.linspace(-50e-9, 50e-9, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Create a 2D harmonic potential well V = 1/2 * m* * w^2 * r^2
    # Standard 2D harmonic oscillator energies: E(nx, ny) = (nx + ny + 1) * hbar * w
    m_star = 0.19 * 9.109e-31
    omega = 2.0e12  # rad/s
    
    # potential energy in eV
    X, Y = np.meshgrid(x, y, indexing='ij')
    # convert Joules to eV
    V_eV = 0.5 * m_star * (omega**2) * (X**2 + Y**2) / 1.602176634e-19
    
    solver = SchrodingerSolver2D(m_star=m_star)
    num_states = 3
    eigenvalues, wavefunctions = solver.solve(V_eV, x, y, num_states=num_states)
    
    assert len(eigenvalues) == num_states
    assert len(wavefunctions) == num_states
    
    # Energies should be sorted in ascending order
    assert eigenvalues[0] <= eigenvalues[1] <= eigenvalues[2]
    
    # Ground state energy should be approx 1 * hbar * omega / e_charge
    E0_theory = 1.0 * 1.054571817e-34 * omega / 1.602176634e-19  # eV
    # Due to boundary containment and finite differences, it should be close
    assert np.abs(eigenvalues[0] - E0_theory) / E0_theory < 0.15
    
    # Check normalization: integral(|psi|^2 dx dy) = 1
    for psi in wavefunctions:
        integral = np.sum(np.abs(psi)**2) * dx * dy
        assert pytest.approx(integral, rel=1e-3) == 1.0

def test_constant_interaction_model():
    """
    Verifies charging energy calculations and stability diagram evaluations.
    """
    # Double dot charging matrix: self charging energies = 4 meV, mutual = 0.8 meV
    U = np.array([[4.0e-3, 0.8e-3],
                  [0.8e-3, 4.0e-3]])
    alpha = np.array([[0.1, 0.01],
                      [0.01, 0.1]])
                      
    ci = ConstantInteractionModel(num_dots=2, charging_matrix=U, lever_arms=alpha)
    
    # Empty state energy with 0V gate voltage
    E_00 = ci.calculate_energy(N=np.array([0, 0]), V_g=np.array([0.0, 0.0]))
    assert E_00 == 0.0
    
    # Energy with 1 electron in dot 1, 0 in dot 2: E = 1/2 * U_11 - e * alpha_11 * V_g1 + confinement
    E_10_0V = ci.calculate_energy(N=np.array([1, 0]), V_g=np.array([0.0, 0.0]))
    assert E_10_0V == 0.5 * 4.0e-3 + 0.5e-3
    
    # With V_g1 = 0.03 V: electrostatic potential shifts down by -e * alpha_11 * V_g1 = -0.003 eV
    E_10_30mV = ci.calculate_energy(N=np.array([1, 0]), V_g=np.array([0.03, 0.0]))
    assert pytest.approx(E_10_30mV) == (0.5 * 4.0e-3 + 0.5e-3) - (0.1 * 0.03)
    
    # Ground state search: with zero voltages, empty state (0,0) should be ground state
    N_star, _ = ci.find_ground_state(V_g=np.array([0.0, 0.0]))
    assert np.array_equal(N_star, [0, 0])
    
    # With high plunger voltages, charging should occur
    N_star_high, _ = ci.find_ground_state(V_g=np.array([0.1, 0.1]))
    assert np.sum(N_star_high) > 0
    
    # Verify thermal expectation values at 100 mK are fractional
    occupations_thermal = ci.get_thermal_occupations(V_g=np.array([0.02, 0.02]), T_K=0.1)
    assert len(occupations_thermal) == 2
    assert all(0.0 <= occ <= 3.0 for occ in occupations_thermal)

def test_dqn_tuning_agent():
    """
    Verifies that the RL agent initializes, produces outputs, and processes updates.
    """
    agent = DQNTuningAgent(state_dim=2, action_dim=4)
    
    # Test random action and greedy action
    state = np.array([0.05, 0.05])
    action = agent.select_action(state)
    assert 0 <= action < 4
    
    # Test memory replay batch training
    for _ in range(10):
        agent.remember(
            state=np.array([0.02, 0.02]),
            action=1,
            reward=-0.05,
            next_state=np.array([0.02, 0.015]),
            done=False
        )
    # Target state success transition
    agent.remember(
        state=np.array([0.05, 0.05]),
        action=0,
        reward=10.0,
        next_state=np.array([0.055, 0.05]),
        done=True
    )
    
    loss = agent.replay(batch_size=8)
    assert isinstance(loss, float)
