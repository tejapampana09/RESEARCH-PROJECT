# QuantumTwin: Scientific Guide & Mathematical Documentation

## 1. Introduction
**QuantumTwin** is an integrated, physics-informed digital twin platform for semiconductor spin qubits in silicon-germanium ($\text{Si/SiGe}$) quantum dot heterostructures. It couples real-time 2D electrostatic potential calculations, sparse Schrödinger eigensolving, many-body Fermi-Hubbard diagonalization, dispersively coupled RF reflectometry, and multi-component noise processes.

---

## 2. Electrostatic Potential Landscape
The potential energy landscape $V(x, y)$ (in eV) experienced by conduction band electrons is modeled as:
\[
V(x, y) = V_{\text{conf}}(x, y) + V_{\text{gates}}(x, y)
\]

### Structural Confinement Well
Each quantum dot $d$ centered at $(x_d, y_d)$ has a Gaussian confinement profile:
\[
V_{\text{conf}, d}(x, y) = -V_{\text{depth}, d} \exp\left( -\frac{(x - x_d)^2 + (y - y_d)^2}{2 w_d^2} \right)
\]
Near the bottom of the well, Taylor expansion yields a 2D isotropic harmonic oscillator with effective spring constant:
\[
k_d = \frac{V_{\text{depth}, d}}{w_d^2} = m^* \omega_d^2 \implies \hbar \omega_d = \hbar \sqrt{\frac{V_{\text{depth}, d}}{m^* w_d^2}}
\]
where $m^* \approx 0.19 m_e$ is the transverse effective mass of electrons in Silicon.

### Control Gate Potential Footprint
Gate electrodes apply electrostatic voltage shifts:
\[
V_{\text{gates}}(x, y) = -e \sum_{k} \alpha_k V_k \exp\left( -\frac{(x - x_k)^2 + (y - y_k)^2}{2 \sigma_k^2} \right)
\]
where $V_k$ is the applied gate voltage, $\alpha_k$ is the lever arm, and $\sigma_k$ is the screening length.

---

## 3. 2D Finite-Difference Schrödinger Eigensolver
The 2D time-independent Schrödinger equation is:
\[
\left( -\frac{\hbar^2}{2 m^*} \nabla^2 + V(x, y) \right) \psi_n(x, y) = E_n \psi_n(x, y)
\]

Discretized on a uniform grid $(N_x, N_y)$ with spacing $dx, dy$, the 2D Laplacian operator $L$ is formed via Kronecker sums:
\[
L = D_{xx} \otimes I_{N_y} + I_{N_x} \otimes D_{yy}
\]
where $D_{xx}$ is the 1D second-derivative tridiagonal matrix $\frac{1}{dx^2}[1, -2, 1]$. Dirichlet boundary conditions are enforced ($\psi = 0$ at domain edges).

The sparse Hamiltonian $H = -\frac{\hbar^2}{2 m^*} L + \text{diag}(V(x, y))$ is diagonalized via `scipy.sparse.linalg.eigsh` in shift-invert mode to obtain normalized wavefunctions $\iint |\psi_n|^2 dx dy = 1$.

---

## 4. Multi-Site Fermi-Hubbard Model
The Fermi-Hubbard Hamiltonian in second-quantized notation is:
\[
H_{\text{Hubbard}} = \sum_{i, \sigma} \epsilon_i n_{i\sigma} + \sum_i U_i n_{i\uparrow} n_{i\downarrow} + \sum_{i < j} U_{ij} n_i n_j - \sum_{\langle i, j \rangle, \sigma} t_{ij}(V_b) \left( c_{i\sigma}^\dagger c_{j\sigma} + \text{H.c.} \right)
\]

For a double quantum dot with 2 electrons, diagonalizing $H_{\text{Hubbard}}$ in the singlet-triplet subspace $\{|S(2,0)\rangle, |S(1,1)\rangle, |S(0,2)\rangle, |T(1,1)\rangle\}$ yields the exact singlet-triplet exchange splitting:
\[
J(\varepsilon) = E_{T_0} - E_{S_0}(\varepsilon)
\]
In the perturbative limit ($t \ll U - U_{12}$ and $\varepsilon = 0$):
\[
J \approx \frac{4 t^2}{U - U_{12}}
\]

---

## 5. RF Reflectometry & Quantum Capacitance
The impedance of an LC tank circuit ($L_p, C_p, R_p$) dispersively coupled to a plunger gate is:
\[
Z_p(\omega) = \frac{1}{\frac{1}{R_p} + i \omega (C_p + C_q) + \frac{1}{i \omega L_p}}
\]
where the quantum capacitance $C_q$ represents capacitive loading due to coherent electron tunneling:
\[
C_q = -e^2 \frac{\partial^2 E_0}{\partial V_g^2}
\]
The complex reflection coefficient $\Gamma = \frac{Z_p - Z_0}{Z_p + Z_0}$ demodulates into In-phase ($I$) and Quadrature ($Q$) components:
\[
I = V_{\text{in}} |\Gamma| \cos(\Delta\theta), \quad Q = V_{\text{in}} |\Gamma| \sin(\Delta\theta)
\]

---

## 6. Multi-Component Noise Architecture
1. **Johnson-Nyquist Noise**: Thermal white noise $S_V = 4 k_B T R$.
2. **Random Telegraph Noise**: Lorentzian spectrum $S(f) = \frac{4 A^2 \bar{\tau}}{1 + (2\pi f \bar{\tau})^2}$.
3. **1/f Pink Noise**: Spectral synthesis $S(f) \propto 1/f^\alpha$ with Welch PSD verification.
4. **Ornstein-Uhlenbeck Drift**: Stationary random walk $d(\delta V) = -\frac{\delta V}{\tau} dt + \sigma dW$.

---

## References
1. F. A. Zwanenburg et al., Rev. Mod. Phys. 85, 961 (2013).
2. W. G. van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002).
3. J. I. Colless et al., Phys. Rev. Lett. 110, 046805 (2013).
4. E. Paladino et al., Rev. Mod. Phys. 86, 361 (2014).
