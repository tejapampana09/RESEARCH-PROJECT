"""
2D Finite-Difference Schrödinger Solver for Quantum Dots
=========================================================

This module solves the 2D time-independent Schrödinger equation for an arbitrary
electrostatic potential landscape V(x, y).

Mathematical Model
------------------
The Schrödinger equation is:
    [ - (hbar^2 / 2m*) \\nabla^2 + V(x, y) ] \\psi_n(x, y) = E_n \\psi_n(x, y)

We discretize this on a uniform 2D grid of size Nx x Ny with grid spacing dx and dy.
The Laplacian \\nabla^2 is represented using a Kronecker sum of 1D second-derivative
finite-difference matrices with Dirichlet boundary conditions (wavefunction vanishes
at grid boundaries):

    L = D_xx \\otimes I_y + I_x \\otimes D_yy

where:
    D_xx is the Nx x Nx tridiagonal matrix with [1, -2, 1] / dx^2
    D_yy is the Ny x Ny tridiagonal matrix with [1, -2, 1] / dy^2
    I_x, I_y are identity matrices.

The Hamiltonian matrix H is:
    H = - (hbar^2 / 2m*) * L + diag(V_flat)

We solve the sparse eigenvalue problem:
    H \\psi_n = E_n \\psi_n
for the lowest K eigenvalues using the Lanczos algorithm via `scipy.sparse.linalg.eigsh`.

Normalization
-------------
The wavefunctions are normalized such that the spatial probability density integrates to 1:
    \\iint |\\psi_n(x, y)|^2 dx dy = 1
    \\sum_{i,j} |\\psi_n(x_i, y_j)|^2 * dx * dy = 1

References
----------
1. S. E. Laux et al., "Semiconductor device simulation of quantum dots," IBM Journal of Research
   and Development 46, 359-380 (2002).
2. J. R. Harrison et al., "Quantum mechanics of nanostructures," (2005).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, List

# Physical Constants (SI Units)
HBAR = 1.054571817e-34      # Reduced Planck's constant (J*s)
E_CHARGE = 1.602176634e-19  # Electron charge magnitude (C or J/eV)
M_E = 9.10938356e-31        # Free electron mass (kg)

class SchrodingerSolver2D:
    """
    Finite-difference solver for the 2D Schrödinger equation on a spatial grid.
    """
    def __init__(self, m_star: float = 0.19 * M_E):
        """
        Args:
            m_star: Effective mass of the electron in the semiconductor (kg).
                    Defaults to the transverse effective mass of silicon (0.19 * m_e).
        """
        self.m_star = m_star

    def solve(self, V_eV: np.ndarray, x: np.ndarray, y: np.ndarray, 
              num_states: int = 6) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solves the Schrödinger equation for the given potential landscape.
        
        Args:
            V_eV: 2D numpy array of shape (Nx, Ny) containing the potential in eV.
            x: 1D array of X coordinates (meters).
            y: 1D array of Y coordinates (meters).
            num_states: Number of lowest eigenstates to solve for.
            
        Returns:
            eigenvalues: 1D array of shape (num_states,) with eigenvalues in eV.
            wavefunctions: List of length num_states containing 2D arrays of shape (Nx, Ny)
                           representing the normalized wavefunctions.
        """
        Nx = len(x)
        Ny = len(y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # 1. Construct 1D second-derivative matrices with Dirichlet boundary conditions
        # Main diagonal contains -2/dx^2, sub/superdiagonals contain 1/dx^2
        dx_diag = np.ones(Nx)
        Dxx = sp.diags([dx_diag[:-1], -2.0 * dx_diag, dx_diag[:-1]], [-1, 0, 1]) / (dx**2)
        
        dy_diag = np.ones(Ny)
        Dyy = sp.diags([dy_diag[:-1], -2.0 * dy_diag, dy_diag[:-1]], [-1, 0, 1]) / (dy**2)
        
        # 2. Construct 2D Laplacian using Kronecker tensor products
        # L = Dxx tensor I_y + I_x tensor Dyy
        L = sp.kron(Dxx, sp.identity(Ny), format='csr') + sp.kron(sp.identity(Nx), Dyy, format='csr')
        
        # Kinetic energy matrix (in Joules)
        T = - (HBAR**2 / (2.0 * self.m_star)) * L
        
        # Potential energy matrix (convert V_eV to Joules)
        V_J = V_eV.ravel() * E_CHARGE
        V_diag = sp.diags(V_J, 0, format='csr')
        
        # Total Hamiltonian
        H = T + V_diag
        
        # 3. Solve for lowest eigenvalues and eigenvectors
        # Shift-invert mode (LM with sigma) is highly stable and accurately resolves degenerate states
        sigma_J = np.min(V_J) - 0.1 * E_CHARGE
        try:
            eigenvalues_J, eigenvectors = spla.eigsh(H, k=num_states, sigma=sigma_J, which='LM')
        except (spla.ArpackNoConvergence, ValueError):
            # Fallback if shift-invert fails or encounters singularity
            print("Shift-invert did not converge, trying standard SM solver...")
            eigenvalues_J, eigenvectors = spla.eigsh(H, k=num_states, which='SM')
            
        # Convert eigenvalues to eV
        eigenvalues_eV = eigenvalues_J / E_CHARGE
        
        # Sort states in ascending energy order (eigsh usually sorts, but we make sure)
        idx = np.argsort(eigenvalues_eV)
        eigenvalues_eV = eigenvalues_eV[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 4. Normalize and reshape wavefunctions
        wavefunctions = []
        for i in range(num_states):
            psi_flat = eigenvectors[:, i]
            # Reshape back to 2D
            psi_2d = psi_flat.reshape((Nx, Ny))
            
            # Normalize: sum(|psi|^2 * dx * dy) = 1
            norm_val = np.sqrt(np.sum(np.abs(psi_2d)**2) * dx * dy)
            psi_normalized = psi_2d / norm_val
            
            # Align phase: make the maximum value positive for visualization consistency
            max_idx = np.unravel_index(np.argmax(np.abs(psi_normalized)), psi_normalized.shape)
            if psi_normalized[max_idx] < 0:
                psi_normalized *= -1.0
                
            wavefunctions.append(psi_normalized)
            
        return eigenvalues_eV, wavefunctions
