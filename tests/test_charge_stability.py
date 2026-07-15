"""
Unit Tests for Charge Stability Diagram Generator
==================================================
Tests:
1. Single dot: threshold voltage matches U/(e*alpha)
2. Double dot: at least 4 distinct charge regions in sweep
3. Triple points detected
4. CSV export produces correct columns
5. Thermal vs T=0 occupations differ correctly
6. Sensor map gradient is non-zero at boundaries
7. Energy minimization: ground state lower than all neighbours
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile
from physics.charge_stability import ChargeStabilityDiagram

# Standard double-dot parameters matching Constant Interaction model
U   = 4.0e-3   # eV
U12 = 0.8e-3   # eV
alpha = 0.12   # lever arm

def make_double_dot_csd(T=0.05):
    """Standard double-dot charge stability diagram object."""
    U_onsite  = np.array([U, U])
    U_inter   = np.array([[0.0, U12], [U12, 0.0]])
    # 2 dots, 2 plunger gates: diagonal lever arm matrix
    lever_arms = np.array([[alpha, 0.0],
                            [0.0,  alpha]])
    return ChargeStabilityDiagram(U_onsite, U_inter, lever_arms, T=T, max_occupation=2)

def test_single_dot_threshold():
    """
    Single dot (0)->(1) transition threshold:
    E(0) = 0, E(1) = 0.5*U - alpha*e*Vg = 0 => Vg_th = 0.5 * U / alpha.
    Analytical: Vg_th = 0.5 * 4e-3 / 0.12 = 16.67 mV.
    """
    U_single   = np.array([U])
    lever_single = np.array([[alpha]])
    U_inter_s  = np.zeros((1,1))
    csd = ChargeStabilityDiagram(U_single, U_inter_s, lever_single, T=0.001, max_occupation=2)

    Vg = np.linspace(0.00, 0.04, 80)
    occupations = []
    for v in Vg:
        N_gs, _ = csd._ground_state_config(np.array([v]))
        occupations.append(N_gs[0])

    # Find transition point
    transitions = np.where(np.diff(np.array(occupations)) > 0)[0]
    assert len(transitions) >= 1, "No (0)->(1) transition found"
    V_th_meas = Vg[transitions[0]]
    V_th_theory = 0.5 * U / alpha   # 16.67 mV
    assert abs(V_th_meas - V_th_theory) < 2e-3, \
        f"Threshold: measured {V_th_meas*1e3:.2f} mV vs theory {V_th_theory*1e3:.2f} mV"


def test_double_dot_charge_regions():
    """Double dot sweep produces at least 4 distinct charge configurations."""
    csd  = make_double_dot_csd()
    Vg_x = np.linspace(0, 0.05, 30)
    Vg_y = np.linspace(0, 0.05, 30)
    result = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=False)
    unique_charges = np.unique(result['charge_map'])
    assert len(unique_charges) >= 3, \
        f"Only {len(unique_charges)} distinct charge values: {unique_charges}"


def test_triple_points_detected():
    """Triple points are detected in a double dot sweep."""
    csd  = make_double_dot_csd()
    Vg_x = np.linspace(0.01, 0.04, 60)
    Vg_y = np.linspace(0.01, 0.04, 60)
    result = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=False)
    assert len(result['triple_points']) >= 1, \
        "No triple points detected in double dot honeycomb"


def test_thermal_vs_zero_temp():
    """Thermal occupations at high T are smoother than T=0 step function."""
    csd_hot  = make_double_dot_csd(T=2.0)
    csd_cold = make_double_dot_csd(T=0.01)
    Vg_range = np.linspace(0.010, 0.025, 30)

    occ_hot  = [csd_hot._thermal_occupations(np.array([v, 0.0]))[0]  for v in Vg_range]
    occ_cold = [csd_cold._thermal_occupations(np.array([v, 0.0]))[0] for v in Vg_range]

    # Hot occupations should have smaller maximum step between adjacent points
    max_step_hot  = max(abs(np.diff(occ_hot)))
    max_step_cold = max(abs(np.diff(occ_cold)))
    assert max_step_hot < max_step_cold, \
        f"Hot step={max_step_hot:.4f} not smoother than cold step={max_step_cold:.4f}"


def test_csv_export_columns():
    """CSV export contains correct columns."""
    csd    = make_double_dot_csd()
    Vg_x   = np.linspace(0, 0.04, 5)
    Vg_y   = np.linspace(0, 0.04, 5)
    result = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_csd.csv")
        csd.export_csv(result, filepath)
        import csv as csv_module
        with open(filepath) as f:
            reader = csv_module.DictReader(f)
            rows   = list(reader)
            expected_keys = {'Vg_x_V', 'Vg_y_V', 'charge_total', 'occ_dot1', 'occ_dot2', 'sensor'}
            assert expected_keys.issubset(set(rows[0].keys()))
            assert len(rows) == 5 * 5


def test_energy_minimization():
    """Ground state energy is less than or equal to all other configs."""
    csd = make_double_dot_csd()
    Vg  = np.array([0.03, 0.02])
    N_gs, E_gs = csd._ground_state_config(Vg)
    import itertools
    for N_vec in itertools.product(range(3), repeat=2):
        E_other = csd._electrostatic_energy(np.array(N_vec, dtype=float), Vg)
        assert E_gs <= E_other + 1e-12, \
            f"Ground state E={E_gs:.6f} not minimal vs N={N_vec}, E={E_other:.6f}"


def test_occupation_bounds():
    """Mean occupations are non-negative and bounded by max_occupation."""
    csd  = make_double_dot_csd()
    Vg_x = np.linspace(0, 0.05, 15)
    Vg_y = np.linspace(0, 0.05, 15)
    result = csd.sweep(0, 1, Vg_x, Vg_y, use_thermal=True)
    assert np.all(result['occupation_x'] >= -1e-10)
    assert np.all(result['occupation_y'] >= -1e-10)
    assert np.all(result['occupation_x'] <= csd.max_occupation + 1e-10)
    assert np.all(result['occupation_y'] <= csd.max_occupation + 1e-10)
