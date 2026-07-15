"""
Fermi-Hubbard Hamiltonian for Silicon Quantum Dot Arrays
=========================================================

Implements a full second-quantized Fermi-Hubbard model for N-site quantum dot
arrays using occupation-number (Fock) basis representation with fixed particle
number sectors.

Physical Model
--------------
H = Σ_{i,σ} ε_i * n_{i,σ}
  + Σ_i   U_i  * n_{i,↑} * n_{i,↓}
  + Σ_{i<j} U_ij * n_i * n_j
  - Σ_{<i,j>,σ} t_ij * (c†_{i,σ} c_{j,σ} + H.c.)

where:
    ε_i    : On-site detuning energy (eV), controlled by plunger gate voltage
    U_i    : On-site charging energy (eV)
    U_ij   : Inter-dot charging energy (eV)
    t_ij   : Tunnel coupling (eV), voltage-dependent: t_ij(V_b) = t0 * exp(γ * V_b)
    n_i    : Total occupation of dot i = n_{i,↑} + n_{i,↓}

Fock Space Representation
--------------------------
For N dots, spin-orbitals are ordered as:
    [0↑, 0↓, 1↑, 1↓, ..., (N-1)↑, (N-1)↓]

Each basis state is a binary integer where bit k is 1 if spin-orbital k is occupied.
Fermion anticommutation signs are tracked via Jordan-Wigner strings.

References
----------
1. W. G. van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002).
2. J. R. Petta et al., Science 309, 2180 (2005) — singlet-triplet qubit.
3. M. A. Eriksson et al., MRS Bulletin 38, 794 (2013) — Si spin qubits.
4. D. Loss and D. P. DiVincenzo, Phys. Rev. A 57, 120 (1998) — exchange J gate.
"""

import logging
import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def _popcount(x: int) -> int:
    """Returns the number of set bits (population count) in integer x."""
    return bin(x).count('1')


def _jordan_wigner_sign(state: int, i: int, j: int) -> int:
    """
    Computes the total Jordan-Wigner phase for the hop c†_i c_j acting on |state⟩.

    The sign is (-1)^(number of occupied orbitals strictly between
    min(i, j) and max(i, j) in |state⟩).  For adjacent orbitals
    (|i - j| == 1) there are no intermediate sites so the sign is +1.

    Args:
        state: Integer representing the Fock state *before* the hop.
        i: Target (creation) orbital index.
        j: Source (annihilation) orbital index.

    Returns:
        +1 or -1.
    """
    lo, hi = min(i, j), max(i, j)
    # No intermediate orbitals → sign is always +1
    if hi - lo <= 1:
        return 1
    # Build mask for bits strictly between lo and hi
    mask = ((1 << (hi - lo - 1)) - 1) << (lo + 1)
    return (-1) ** _popcount(state & mask)


def _build_fock_basis(n_sites: int, n_electrons: int) -> List[int]:
    """
    Constructs all Fock basis states for n_electrons in n_sites dots (2*n_sites spin-orbitals).

    Returns:
        Sorted list of integer state representations.
    """
    n_orbitals = 2 * n_sites
    states = [
        sum(1 << k for k in combo)
        for combo in combinations(range(n_orbitals), n_electrons)
    ]
    return sorted(states)


class FermiHubbardModel:
    """
    Fermi-Hubbard Hamiltonian for an N-site quantum dot array.

    Constructs and diagonalizes the full many-body Hamiltonian in the
    fixed-particle-number sector using occupation-number basis representation.

    Parameters
    ----------
    n_sites : int
        Number of quantum dots.
    U_onsite : array_like of float
        On-site charging energies U_i (eV), shape (n_sites,).
    U_inter : array_like of float
        Inter-dot charging energies U_ij (eV), shape (n_sites, n_sites).
        Only upper triangle is used.
    t_matrix : array_like of float
        Tunnel coupling matrix t_ij (eV), shape (n_sites, n_sites).
        Symmetric, zero diagonal.
    epsilon : array_like of float
        On-site detuning energies ε_i (eV), shape (n_sites,).
        Controlled by plunger gate voltages.
    n_electrons : int
        Total number of electrons in the system.
    """

    def __init__(self,
                 n_sites: int,
                 U_onsite: np.ndarray,
                 U_inter: np.ndarray,
                 t_matrix: np.ndarray,
                 epsilon: np.ndarray,
                 n_electrons: int):
        self.n_sites = n_sites
        self.n_orbitals = 2 * n_sites     # spin-up and spin-down per site
        self.U_onsite = np.asarray(U_onsite, dtype=float)
        self.U_inter  = np.asarray(U_inter,  dtype=float)
        self.t_matrix = np.asarray(t_matrix, dtype=float)
        self.epsilon  = np.asarray(epsilon,  dtype=float)
        self.n_electrons = n_electrons

        # Build the Fock basis for fixed particle number
        self.basis = _build_fock_basis(n_sites, n_electrons)
        self.dim   = len(self.basis)
        self._state_to_idx: Dict[int, int] = {s: i for i, s in enumerate(self.basis)}

        logger.info(
            f"FermiHubbardModel: {n_sites} sites, {n_electrons} electrons, "
            f"Hilbert space dim = {self.dim}"
        )

    def _orbital(self, site: int, spin: int) -> int:
        """
        Returns the spin-orbital index for a given site and spin.

        Spin-orbital ordering: [site0↑, site0↓, site1↑, site1↓, ...]

        Args:
            site: Dot index (0-indexed).
            spin: 0 for spin-up, 1 for spin-down.
        """
        return 2 * site + spin

    def build_hamiltonian(self) -> np.ndarray:
        """
        Constructs the full Hamiltonian matrix H in the Fock basis.

        Matrix elements are computed as:
            H[α, β] = ⟨α| H |β⟩

        Returns:
            H: Dense Hermitian Hamiltonian matrix of shape (dim, dim).
        """
        H = np.zeros((self.dim, self.dim), dtype=float)

        for beta_idx, state_b in enumerate(self.basis):

            # ── Diagonal terms ─────────────────────────────────────────────
            # On-site detuning: Σ_{i,σ} ε_i * n_{i,σ}
            # On-site Coulomb:  Σ_i U_i * n_{i,↑} * n_{i,↓}
            # Inter-dot:        Σ_{i<j} U_ij * n_i * n_j
            diag = 0.0
            for i in range(self.n_sites):
                orb_up   = self._orbital(i, 0)
                orb_down = self._orbital(i, 1)
                n_up_i   = (state_b >> orb_up)   & 1
                n_down_i = (state_b >> orb_down)  & 1
                n_i      = n_up_i + n_down_i

                # Detuning (same energy for both spins on the same dot)
                diag += self.epsilon[i] * n_i
                # On-site Coulomb repulsion
                diag += self.U_onsite[i] * n_up_i * n_down_i

            for i in range(self.n_sites):
                for j in range(i + 1, self.n_sites):
                    n_i = ((state_b >> self._orbital(i, 0)) & 1) + \
                          ((state_b >> self._orbital(i, 1)) & 1)
                    n_j = ((state_b >> self._orbital(j, 0)) & 1) + \
                          ((state_b >> self._orbital(j, 1)) & 1)
                    diag += self.U_inter[i, j] * n_i * n_j

            H[beta_idx, beta_idx] = diag

            # ── Off-diagonal hopping terms ─────────────────────────────────
            # -t_ij * (c†_{i,σ} c_{j,σ} + H.c.) for each bond <i,j> and spin σ
            for i in range(self.n_sites):
                for j in range(self.n_sites):
                    if i == j:
                        continue
                    t = self.t_matrix[i, j]
                    if abs(t) < 1e-15:
                        continue

                    for spin in range(2):
                        orb_i = self._orbital(i, spin)
                        orb_j = self._orbital(j, spin)

                        # Attempt c†_{i,σ} c_{j,σ} |β⟩:
                        # 1) orbital j must be occupied in |β⟩
                        # 2) orbital i must be empty in |β⟩
                        if not (state_b >> orb_j) & 1:
                            continue
                        if (state_b >> orb_i) & 1:
                            continue

                        # New state: remove electron from orb_j, place at orb_i
                        new_state = state_b ^ (1 << orb_j) ^ (1 << orb_i)

                        if new_state not in self._state_to_idx:
                            continue

                        alpha_idx = self._state_to_idx[new_state]

                        # Total Jordan-Wigner phase for c†_{orb_i} c_{orb_j} |state_b⟩
                        # = (-1)^(# occupied orbitals strictly between orb_i and orb_j)
                        sign = _jordan_wigner_sign(state_b, orb_i, orb_j)

                        # H matrix element: -t_ij * JW_sign
                        H[alpha_idx, beta_idx] += -t * sign

        return H

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalizes the Hamiltonian and returns eigenvalues and eigenvectors.

        Returns:
            eigenvalues: Array of sorted eigenvalues (eV), shape (dim,).
            eigenvectors: Corresponding eigenvectors, shape (dim, dim).
        """
        H = self.build_hamiltonian()
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def get_sz_expectation(self, eigenvector: np.ndarray) -> float:
        """
        Computes ⟨S_z⟩ = Σ_α |⟨α|ψ⟩|² * Sz(α) for a given eigenvector.

        Args:
            eigenvector: Normalized eigenvector in the Fock basis.

        Returns:
            S_z expectation value (in units of ħ/2).
        """
        sz = 0.0
        for idx, state in enumerate(self.basis):
            sz_state = 0.0
            for i in range(self.n_sites):
                n_up   = (state >> self._orbital(i, 0)) & 1
                n_down = (state >> self._orbital(i, 1)) & 1
                sz_state += 0.5 * (n_up - n_down)
            sz += (abs(eigenvector[idx])**2) * sz_state
        return sz

    def get_double_occupation(self, eigenvector: np.ndarray) -> np.ndarray:
        """
        Returns the average double occupation ⟨n_{i↑} n_{i↓}⟩ per dot.

        Args:
            eigenvector: Normalized eigenvector.

        Returns:
            Array of double occupancy probabilities, shape (n_sites,).
        """
        d_occ = np.zeros(self.n_sites)
        for idx, state in enumerate(self.basis):
            prob = abs(eigenvector[idx])**2
            for i in range(self.n_sites):
                n_up   = (state >> self._orbital(i, 0)) & 1
                n_down = (state >> self._orbital(i, 1)) & 1
                d_occ[i] += prob * n_up * n_down
        return d_occ


class DoubleDotST:
    """
    Specialized singlet-triplet analysis for a 2-site (double dot) system.

    Provides analytical and numerical exchange splitting J = E_T - E_S
    as a function of detuning ε and tunnel coupling t.

    The singlet manifold has 3 basis states:
        |S(2,0)⟩, |S(1,1)⟩, |S(0,2)⟩

    The triplet T_0 state does not mix with singlets (different S_z sector for T±,
    and no hopping connects T_0 to singlets at fixed S_z = 0).

    Parameters
    ----------
    U1 : float
        On-site charging energy of dot 1 (eV).
    U2 : float
        On-site charging energy of dot 2 (eV).
    U12 : float
        Inter-dot charging energy (eV).
    t0 : float
        Base tunnel coupling at V_b = 0 (eV).
    gamma : float
        Barrier lever arm for exponential t(V_b) = t0 * exp(gamma * V_b) (1/V).
    """

    def __init__(self,
                 U1:    float = 4.0e-3,
                 U2:    float = 4.0e-3,
                 U12:   float = 0.8e-3,
                 t0:    float = 50e-6,
                 gamma: float = 20.0):
        self.U1    = U1
        self.U2    = U2
        self.U12   = U12
        self.t0    = t0
        self.gamma = gamma

    def tunnel_coupling(self, V_b: float) -> float:
        """
        Returns the voltage-dependent tunnel coupling.

        t(V_b) = t0 * exp(gamma * V_b)

        Args:
            V_b: Barrier gate voltage (V).
        """
        return self.t0 * np.exp(self.gamma * V_b)

    def build_singlet_hamiltonian(self, epsilon: float, t: float) -> np.ndarray:
        """
        Constructs the 3×3 Hamiltonian in the singlet subspace.

        Basis: {|S(2,0)⟩, |S(1,1)⟩, |S(0,2)⟩}

        Energies:
            E[S(2,0)] = U1 - ε   (double occupancy on dot 1)
            E[S(1,1)] = U12       (one electron per dot, inter-dot repulsion)
            E[S(0,2)] = U2 + ε   (double occupancy on dot 2)

        The √2 factor in tunnel matrix elements arises from the two-particle
        overlap ⟨S(1,1)|H_t|S(2,0)⟩ = -√2 * t (Clebsch-Gordan coefficient
        for combining two spin-1/2 particles into a singlet).

        Args:
            epsilon: Detuning ε = ε_1 - ε_2 (eV). Positive ε favors |S(2,0)⟩.
            t: Tunnel coupling (eV).

        Returns:
            3×3 Hamiltonian matrix (eV).
        """
        sqrt2_t = np.sqrt(2.0) * t
        H_S = np.array([
            [self.U1 - epsilon,  -sqrt2_t,           0.0          ],
            [-sqrt2_t,            self.U12,           -sqrt2_t     ],
            [0.0,                 -sqrt2_t,           self.U2 + epsilon]
        ])
        return H_S

    def compute_exchange(self,
                         epsilon: float,
                         V_b:     float = 0.0) -> Dict[str, float]:
        """
        Computes the singlet-triplet exchange splitting J = E_{T_0} - E_{S_0}.

        The triplet T_0 energy (Sz=0) in the (1,1) charge sector is simply U12
        (no Coulomb penalty for double occupation, and no tunnel coupling to
        (2,0) or (0,2) singlets due to spin selection rules).

        Args:
            epsilon: Detuning ε_1 - ε_2 (eV).
            V_b: Barrier gate voltage (V), determines tunnel coupling.

        Returns:
            Dictionary with keys:
                'E_S0'    : Singlet ground state energy (eV)
                'E_T0'    : Triplet T_0 energy (eV)
                'J'       : Exchange splitting E_T0 - E_S0 (eV)
                't'       : Tunnel coupling used (eV)
                'E_singlets': All 3 singlet eigenvalues (eV)
        """
        t = self.tunnel_coupling(V_b)
        H_S = self.build_singlet_hamiltonian(epsilon, t)
        E_singlets = np.linalg.eigvalsh(H_S)
        E_S0 = E_singlets[0]

        # T_0 energy = U12 (no hopping, only inter-dot repulsion)
        E_T0 = self.U12

        J = E_T0 - E_S0

        return {
            'E_S0':      E_S0,
            'E_T0':      E_T0,
            'J':         J,
            't':         t,
            'E_singlets': E_singlets
        }

    def sweep_detuning(self,
                       epsilon_range: np.ndarray,
                       V_b: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Sweeps detuning ε and returns energy spectrum and exchange splitting J(ε).

        Args:
            epsilon_range: Array of detuning values (eV).
            V_b: Barrier gate voltage (V).

        Returns:
            Dictionary with 'epsilon', 'E_singlets' (shape N×3), 'E_T0', 'J'.
        """
        t = self.tunnel_coupling(V_b)
        results = [
            np.linalg.eigvalsh(self.build_singlet_hamiltonian(eps, t))
            for eps in epsilon_range
        ]
        E_singlets = np.array(results)  # shape (N, 3)
        E_T0 = np.full(len(epsilon_range), self.U12)
        J    = E_T0 - E_singlets[:, 0]

        return {
            'epsilon':    epsilon_range,
            'E_singlets': E_singlets,
            'E_T0':       E_T0,
            'J':          J
        }

    def analytical_J_deep_detuning(self, t: Optional[float] = None) -> float:
        """
        Returns the analytical exchange splitting in the (1,1) regime (ε → 0):

            J ≈ 4 * t² / (U_avg - U12)

        where U_avg = (U1 + U2) / 2.

        This is the perturbative result to second order in t/(U - U12).

        Args:
            t: Tunnel coupling (eV). Defaults to self.t0.

        Returns:
            J_analytical (eV).
        """
        if t is None:
            t = self.t0
        U_avg = 0.5 * (self.U1 + self.U2)
        return 4.0 * t**2 / (U_avg - self.U12)


def build_voltage_dependent_t(t0: float,
                               gamma: float,
                               V_b: float) -> float:
    """
    Computes the voltage-dependent tunnel coupling.

    t(V_b) = t0 * exp(gamma * V_b)

    Physical basis: The tunnel barrier height is lowered approximately
    linearly by the barrier gate voltage, causing an exponential change
    in the WKB tunneling transmission through the barrier potential.

    Args:
        t0: Zero-voltage tunnel coupling (eV).
        gamma: Barrier lever arm (1/V). Typical values: 10–50 V⁻¹.
        V_b: Applied barrier gate voltage (V). Negative V_b closes the barrier.

    Returns:
        t (eV).
    """
    return t0 * np.exp(gamma * V_b)
