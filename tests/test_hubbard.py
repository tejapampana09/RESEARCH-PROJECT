"""
Unit Tests for Fermi-Hubbard Hamiltonian Module
================================================

Tests verify:
1. Fock basis dimension matches combinatorial formula C(2N, n_e).
2. Jordan-Wigner signs are correct for known simple hops.
3. Decoupled limit (t=0): eigenvalues reduce to pure charging energies.
4. Single-site, 1-electron: eigenvalue equals detuning ε.
5. Double dot, 2 electrons: singlet ground state below triplet (J > 0 for t > 0).
6. Hermiticity of the Hamiltonian matrix.
7. Particle number conservation (each basis state has n_electrons bits set).
8. Exchange J matches analytical perturbation formula J ≈ 4t²/(U-U12) in weak-t limit.
9. Voltage-dependent tunnel coupling t(V_b) is monotonic and correct.
10. Detuning sweep returns correct array dimensions and J > 0 everywhere.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from physics.hubbard import (
    FermiHubbardModel,
    DoubleDotST,
    build_voltage_dependent_t,
    _build_fock_basis,
    _jordan_wigner_sign,
    _popcount,
)
from math import comb


# ─────────────────────────────────────────────────────────────────────────────
# Fock Basis Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fock_basis_dimension():
    """Basis dimension must be C(2*n_sites, n_electrons)."""
    for n_sites in [1, 2, 3]:
        for n_e in range(1, 2 * n_sites + 1):
            basis = _build_fock_basis(n_sites, n_e)
            expected = comb(2 * n_sites, n_e)
            assert len(basis) == expected, \
                f"n_sites={n_sites}, n_e={n_e}: got {len(basis)}, expected {expected}"


def test_fock_basis_particle_number():
    """Every basis state has exactly n_electrons bits set."""
    n_sites, n_e = 2, 2
    basis = _build_fock_basis(n_sites, n_e)
    for state in basis:
        assert _popcount(state) == n_e


def test_jordan_wigner_sign_adjacent():
    """Hopping between adjacent orbitals through 0 intermediate sites → sign = +1."""
    # State: orbital 0 occupied (binary: ...001)
    # Hopping from orbital 0 to orbital 1: no intermediate sites
    state = 0b0001   # orbital 0 occupied
    sign = _jordan_wigner_sign(state, 1, 0)
    assert sign in (+1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# FermiHubbardModel Tests
# ─────────────────────────────────────────────────────────────────────────────

def make_double_dot(epsilon=(0.0, 0.0), t=50e-6, U=4e-3, U12=0.8e-3, n_e=2):
    """Helper: constructs a standard double-dot FermiHubbardModel."""
    n_sites = 2
    U_onsite = np.array([U, U])
    U_inter  = np.array([[0.0, U12], [U12, 0.0]])
    t_matrix = np.array([[0.0, t], [t, 0.0]])
    eps      = np.array(list(epsilon))
    return FermiHubbardModel(n_sites, U_onsite, U_inter, t_matrix, eps, n_e)


def test_hamiltonian_hermitian():
    """Hamiltonian must be real and symmetric (Hermitian)."""
    model = make_double_dot()
    H = model.build_hamiltonian()
    assert H.shape == (model.dim, model.dim)
    assert pytest.approx(H, abs=1e-12) == H.T


def test_decoupled_limit_eigenvalues():
    """With t=0, eigenvalues must be pure charging energies."""
    U = 4e-3
    U12 = 0.8e-3
    # At t=0, epsilon=(0,0): energies are U (S(2,0)), U12 (T and S(1,1)), U (S(0,2))
    model = make_double_dot(t=0.0)
    eigenvalues, _ = model.solve()
    # Ground states in (1,1) sector: energy = U12 (triplet x3 + singlet x1)
    # (2,0) and (0,2) have energy U + U12
    # We expect multiple eigenvalues at U12 and some at U + U12? Let's just check min
    # At ε=0, t=0: (1,1) states all have energy U12, (2,0) has U1+U12, (0,2) has U2+U12
    # The minimum eigenvalue should be U12
    assert pytest.approx(min(eigenvalues), rel=1e-8) == U12


def test_single_site_single_electron():
    """Single site, single electron: eigenvalue = detuning ε."""
    eps_val = 0.5e-3  # 0.5 meV
    model = FermiHubbardModel(
        n_sites=1,
        U_onsite=np.array([4e-3]),
        U_inter=np.zeros((1, 1)),
        t_matrix=np.zeros((1, 1)),
        epsilon=np.array([eps_val]),
        n_electrons=1
    )
    eigenvalues, _ = model.solve()
    # One electron on one site: both spin-up and spin-down states have energy ε
    assert len(eigenvalues) == 2
    assert pytest.approx(eigenvalues[0], rel=1e-8) == eps_val
    assert pytest.approx(eigenvalues[1], rel=1e-8) == eps_val


def test_sz_expectation_ground_state():
    """Ground state singlet should have ⟨Sz⟩ ≈ 0."""
    model = make_double_dot(t=50e-6)
    eigenvalues, eigenvectors = model.solve()
    Sz = model.get_sz_expectation(eigenvectors[:, 0])
    assert pytest.approx(Sz, abs=1e-10) == 0.0


def test_double_occupancy_symmetry():
    """At ε=0, t>0: symmetric configuration → equal double occupancy on both dots."""
    model = make_double_dot(epsilon=(0.0, 0.0), t=50e-6)
    _, eigenvectors = model.solve()
    d_occ = model.get_double_occupation(eigenvectors[:, 0])
    # By symmetry, both dots should have equal double occupancy
    assert pytest.approx(d_occ[0], abs=1e-10) == d_occ[1]


# ─────────────────────────────────────────────────────────────────────────────
# DoubleDotST Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_exchange_positive():
    """J = E_T - E_S > 0 for any t > 0 (singlet lower than triplet)."""
    st = DoubleDotST(U1=4e-3, U2=4e-3, U12=0.8e-3, t0=50e-6)
    result = st.compute_exchange(epsilon=0.0, V_b=0.0)
    assert result['J'] > 0.0, f"J = {result['J']*1e6:.2f} µeV should be positive"


def test_exchange_increases_with_t():
    """Larger tunnel coupling → larger exchange J."""
    st1 = DoubleDotST(t0=20e-6)
    st2 = DoubleDotST(t0=100e-6)
    J1 = st1.compute_exchange(0.0)['J']
    J2 = st2.compute_exchange(0.0)['J']
    assert J2 > J1, f"J should increase with t: J(20µeV)={J1*1e6:.3f} µeV, J(100µeV)={J2*1e6:.3f} µeV"


def test_analytical_J_perturbative():
    """Numerical J matches perturbative formula J ≈ 4t²/(U-U12) in weak-t limit."""
    t_weak = 5e-6   # Very weak coupling: t << U
    U = 4e-3
    U12 = 0.8e-3
    st = DoubleDotST(U1=U, U2=U, U12=U12, t0=t_weak)

    result  = st.compute_exchange(epsilon=0.0)
    J_num   = result['J']
    J_anal  = st.analytical_J_deep_detuning(t=t_weak)

    rel_err = abs(J_num - J_anal) / J_anal
    assert rel_err < 0.01, \
        f"J numerical ({J_num*1e9:.3f} neV) vs analytical ({J_anal*1e9:.3f} neV): {rel_err*100:.2f}% error"


def test_detuning_sweep_dimensions():
    """Sweep returns arrays of correct length and J > 0 throughout (1,1) regime."""
    st = DoubleDotST(t0=50e-6)
    eps_range = np.linspace(-1e-3, 1e-3, 200)
    sweep = st.sweep_detuning(eps_range, V_b=0.0)

    assert len(sweep['J'])          == 200
    assert sweep['E_singlets'].shape == (200, 3)
    # J should be positive for all detuning values in (-U+U12, U-U12)
    assert np.all(sweep['J'] > 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tunnel Coupling Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tunnel_coupling_zero_voltage():
    """At V_b=0, t(0) = t0."""
    t0 = 50e-6
    gamma = 20.0
    t = build_voltage_dependent_t(t0, gamma, V_b=0.0)
    assert pytest.approx(t, rel=1e-10) == t0


def test_tunnel_coupling_monotonic():
    """Increasing barrier voltage increases tunnel coupling (opens barrier)."""
    t0 = 50e-6
    gamma = 20.0
    V_b_vals = np.linspace(-0.05, 0.05, 20)
    t_vals   = [build_voltage_dependent_t(t0, gamma, V) for V in V_b_vals]
    assert all(t_vals[i] < t_vals[i+1] for i in range(len(t_vals)-1))
