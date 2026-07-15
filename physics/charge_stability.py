"""
Charge Stability Diagram Generator
====================================
Generates publication-quality charge stability (honeycomb) diagrams for
N-dot quantum dot arrays by sweeping plunger gate voltages and computing
the ground-state charge configuration using the Constant Interaction model.

Physical Model
--------------
The ground-state charge configuration N* = argmin_N E(N, Vg) where:
    E(N, Vg) = 0.5 * N^T * U * N - e * N^T * alpha * Vg + sum_i sum_n E_n,i

Honeycomb boundaries occur where two charge configurations are degenerate:
    E(N, Vg) = E(N', Vg)

For a double dot, this produces the well-known honeycomb stability diagram
with triple points at corners of hexagonal cells.

References
----------
1. W. G. van der Wiel et al., Rev. Mod. Phys. 75, 1 (2002).
2. L. DiCarlo et al., Phys. Rev. Lett. 92, 226801 (2004).
3. T. Ihn, Semiconductor Nanostructures, Oxford (2010).
"""

import logging
import numpy as np
import os
from typing import Optional, Tuple, Dict, List
import csv

logger = logging.getLogger(__name__)

# Physical constants
E_CHARGE = 1.602176634e-19  # C
K_B      = 1.380649e-23     # J/K


class ChargeStabilityDiagram:
    """
    Generates charge stability (honeycomb) diagrams for quantum dot arrays
    by sweeping two plunger gate voltages.

    Parameters
    ----------
    U_onsite : np.ndarray
        On-site charging energies U_i (eV), shape (n_dots,).
    U_inter : np.ndarray
        Inter-dot charging energy matrix U_ij (eV), shape (n_dots, n_dots).
        Symmetric, zero diagonal.
    lever_arms : np.ndarray
        Gate lever arm matrix alpha_ij (dimensionless), shape (n_dots, n_gates).
        alpha_ij is the coupling strength of gate j to dot i.
    orbital_energies : np.ndarray
        Single-particle orbital energies E_n,i (eV), shape (n_dots, max_occ).
    T : float
        Electron temperature (K). Determines thermal broadening.
    max_occupation : int
        Maximum electrons per dot to consider.
    """

    def __init__(self,
                 U_onsite:        np.ndarray,
                 U_inter:         np.ndarray,
                 lever_arms:      np.ndarray,
                 orbital_energies: Optional[np.ndarray] = None,
                 T:               float = 0.1,
                 max_occupation:  int   = 3):
        self.U_onsite         = np.asarray(U_onsite, dtype=float)
        self.U_inter          = np.asarray(U_inter,  dtype=float)
        self.lever_arms       = np.asarray(lever_arms, dtype=float)
        self.n_dots           = len(U_onsite)
        self.T                = T
        self.max_occupation   = max_occupation

        if orbital_energies is None:
            # Default: flat orbital spectrum (zero single-particle energies)
            self.orbital_energies = np.zeros((self.n_dots, max_occupation + 1))
        else:
            self.orbital_energies = np.asarray(orbital_energies, dtype=float)

        # Build the charging energy matrix U (eV)
        self.U_matrix = np.diag(self.U_onsite) + self.U_inter
        np.fill_diagonal(self.U_matrix, self.U_onsite)

        logger.info(f"ChargeStabilityDiagram: {self.n_dots} dots, T={T} K, "
                    f"max_occ={max_occupation}")

    def _orbital_energy_sum(self, N_vec: np.ndarray) -> float:
        """Sum of single-particle energies for occupation vector N_vec."""
        total = 0.0
        for i, n in enumerate(N_vec):
            n_int = int(n)
            for k in range(n_int):
                if k < self.orbital_energies.shape[1]:
                    total += self.orbital_energies[i, k]
        return total

    def _electrostatic_energy(self, N_vec: np.ndarray,
                                Vg: np.ndarray) -> float:
        """
        Computes the total electrostatic energy E(N, Vg) in eV.

        E(N, Vg) = 0.5 * N^T U N - e * N^T * alpha * Vg + E_orbital(N)

        Args:
            N_vec : Charge configuration vector (number of electrons per dot).
            Vg    : Gate voltage vector (V).

        Returns:
            Energy in eV.
        """
        N = np.asarray(N_vec, dtype=float)
        # Coulomb term: 0.5 * N^T * U * N
        coulomb = 0.5 * N @ self.U_matrix @ N
        # Gate coupling term: -N^T * alpha * Vg
        gate_coupling = -N @ (self.lever_arms @ Vg)
        # Single-particle energies
        orbital = self._orbital_energy_sum(N_vec)
        return coulomb + gate_coupling + orbital

    def _ground_state_config(self, Vg: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Finds the ground-state charge configuration at gate voltage Vg.

        Exhaustively searches all configurations N with 0 <= N_i <= max_occupation.

        Args:
            Vg : Gate voltage vector (V).

        Returns:
            (N_gs, E_gs): Ground-state occupation vector and energy (eV).
        """
        min_energy = np.inf
        N_gs = np.zeros(self.n_dots, dtype=int)

        # Generate all charge configurations
        ranges = [range(self.max_occupation + 1)] * self.n_dots
        import itertools
        for N_vec in itertools.product(*ranges):
            N_arr = np.array(N_vec, dtype=float)
            E = self._electrostatic_energy(N_arr, Vg)
            if E < min_energy:
                min_energy = E
                N_gs = np.array(N_vec, dtype=int)

        return N_gs, min_energy

    def _thermal_occupations(self, Vg: np.ndarray) -> np.ndarray:
        """
        Computes thermally averaged occupation <N_i> at finite temperature T.

        Uses canonical ensemble over all configurations:
            <N_i> = sum_N N_i * exp(-E(N,Vg)/kT) / Z

        Args:
            Vg : Gate voltage vector (V).

        Returns:
            Mean occupation array, shape (n_dots,).
        """
        import itertools
        kT = K_B * self.T / E_CHARGE   # kT in eV

        energies = []
        configs  = []
        ranges   = [range(self.max_occupation + 1)] * self.n_dots
        for N_vec in itertools.product(*ranges):
            N_arr = np.array(N_vec, dtype=float)
            E     = self._electrostatic_energy(N_arr, Vg)
            energies.append(E)
            configs.append(N_arr)

        energies = np.array(energies)
        # Shift by minimum for numerical stability
        energies -= energies.min()
        weights   = np.exp(-energies / kT)
        Z         = weights.sum()

        mean_N = np.zeros(self.n_dots)
        for w, N_arr in zip(weights, configs):
            mean_N += (w / Z) * N_arr

        return mean_N

    def sweep(self,
              gate_idx_x: int,
              gate_idx_y: int,
              Vg_x: np.ndarray,
              Vg_y: np.ndarray,
              Vg_fixed: Optional[np.ndarray] = None,
              use_thermal: bool = True) -> Dict:
        """
        Sweeps two plunger gates and computes the charge stability diagram.

        Args:
            gate_idx_x  : Index of gate swept along x-axis.
            gate_idx_y  : Index of gate swept along y-axis.
            Vg_x        : Voltage values for gate x (V).
            Vg_y        : Voltage values for gate y (V).
            Vg_fixed    : Fixed voltages for all gates (V). Defaults to zero.
            use_thermal : If True, compute thermal occupations. If False, use T=0.

        Returns:
            Dict with keys:
                'Vg_x', 'Vg_y' : Voltage axes (V)
                'charge_map'   : 2D array of ground-state total charge N_x + N_y
                'occupation_x' : 2D array of <N_0> (first dot)
                'occupation_y' : 2D array of <N_1> (second dot)
                'charge_sensor': 2D simulated sensor signal
                'triple_points': List of (Vx, Vy) triple point coordinates
        """
        n_x   = len(Vg_x)
        n_y   = len(Vg_y)
        n_gates = self.lever_arms.shape[1]

        if Vg_fixed is None:
            Vg_fixed = np.zeros(n_gates)

        charge_map   = np.zeros((n_y, n_x), dtype=int)
        occ_x        = np.zeros((n_y, n_x))
        occ_y        = np.zeros((n_y, n_x))
        sensor_map   = np.zeros((n_y, n_x))

        logger.info(f"Sweeping {n_x}x{n_y} grid...")

        for j, vy in enumerate(Vg_y):
            for i, vx in enumerate(Vg_x):
                Vg = Vg_fixed.copy()
                Vg[gate_idx_x] = vx
                Vg[gate_idx_y] = vy

                if use_thermal:
                    mean_N = self._thermal_occupations(Vg)
                    charge_map[j, i] = int(np.round(mean_N.sum()))
                    occ_x[j, i]      = mean_N[0]
                    occ_y[j, i]      = mean_N[1] if self.n_dots > 1 else 0.0
                else:
                    N_gs, _ = self._ground_state_config(Vg)
                    charge_map[j, i] = N_gs.sum()
                    occ_x[j, i]      = N_gs[0]
                    occ_y[j, i]      = N_gs[1] if self.n_dots > 1 else 0

                # Simulated charge sensor: weighted sum of occupations
                sensor_map[j, i] = occ_x[j, i] + occ_y[j, i]

        # Detect triple points: corners where 3 charge regions meet
        triple_points = self._find_triple_points(charge_map, Vg_x, Vg_y)

        return {
            'Vg_x':         Vg_x,
            'Vg_y':         Vg_y,
            'charge_map':   charge_map,
            'occupation_x': occ_x,
            'occupation_y': occ_y,
            'charge_sensor': sensor_map,
            'triple_points': triple_points,
        }

    def _find_triple_points(self,
                             charge_map: np.ndarray,
                             Vg_x:       np.ndarray,
                             Vg_y:       np.ndarray) -> List[Tuple[float, float]]:
        """
        Finds triple points by locating 2x2 grid intersections where 3 distinct
        charge configurations meet or where the gradient magnitude is localized.

        Returns:
            List of (Vx, Vy) coordinates of detected triple points.
        """
        triple_points = []
        for j in range(1, len(Vg_y) - 1):
            for i in range(1, len(Vg_x) - 1):
                # Check 2x2 neighborhood
                nb = {
                    charge_map[j-1, i-1], charge_map[j-1, i],
                    charge_map[j,   i-1], charge_map[j,   i]
                }
                if len(nb) >= 3:
                    triple_points.append((float(Vg_x[i]), float(Vg_y[j])))

        # Fallback for coarse grids: locate boundary junction points
        if not triple_points:
            diff_x = np.abs(np.diff(charge_map, axis=1))
            diff_y = np.abs(np.diff(charge_map, axis=0))
            for j in range(1, diff_y.shape[0]):
                for i in range(1, diff_x.shape[1]):
                    if diff_x[j, i] > 0 and diff_y[j, i] > 0:
                        triple_points.append((float(Vg_x[i]), float(Vg_y[j])))

        return triple_points

    def export_csv(self, result: Dict, filepath: str) -> None:
        """
        Exports the stability diagram data to a CSV file.

        Columns: Vg_x (V), Vg_y (V), charge_total, occ_x, occ_y, sensor
        """
        rows = []
        for j, vy in enumerate(result['Vg_y']):
            for i, vx in enumerate(result['Vg_x']):
                rows.append({
                    'Vg_x_V':       vx,
                    'Vg_y_V':       vy,
                    'charge_total': result['charge_map'][j, i],
                    'occ_dot1':     result['occupation_x'][j, i],
                    'occ_dot2':     result['occupation_y'][j, i],
                    'sensor':       result['charge_sensor'][j, i],
                })

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported CSV: {filepath} ({len(rows)} rows)")

    def plot(self, result: Dict,
             title: str = "Charge Stability Diagram",
             save_path: Optional[str] = None,
             show_triple_points: bool = True) -> None:
        """
        Generates publication-quality charge stability honeycomb diagram.

        Args:
            result             : Output from self.sweep().
            title              : Figure title.
            save_path          : Base path for PNG/PDF export (without extension).
            show_triple_points : Overlay triple point markers.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
        from matplotlib import ticker

        plt.rcParams.update({
            "font.family": "serif", "font.size": 11,
            "axes.labelsize": 13, "axes.titlesize": 13,
            "savefig.dpi": 300,
        })

        Vg_x = result['Vg_x'] * 1e3  # V → mV
        Vg_y = result['Vg_y'] * 1e3

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

        # ── Panel 1: Charge map ────────────────────────────────────────────
        ax = axes[0]
        charge = result['charge_map']
        n_levels = charge.max() - charge.min() + 2
        cmap = plt.cm.get_cmap('Blues', n_levels)
        bounds = np.arange(charge.min() - 0.5, charge.max() + 1.5)
        norm  = BoundaryNorm(bounds, cmap.N)
        im0   = ax.pcolormesh(Vg_x, Vg_y, charge, cmap=cmap, norm=norm,
                               shading='nearest')
        cbar0 = fig.colorbar(im0, ax=ax, ticks=np.arange(charge.min(), charge.max()+1))
        cbar0.set_label('Total Charge N')
        if show_triple_points and result['triple_points']:
            tp = np.array(result['triple_points']) * 1e3
            ax.scatter(tp[:, 0], tp[:, 1], c='red', s=15, zorder=5,
                       label='Triple points')
            ax.legend(fontsize=9, loc='lower right')
        ax.set_xlabel(r'$V_{g1}$ (mV)')
        ax.set_ylabel(r'$V_{g2}$ (mV)')
        ax.set_title('Charge Map')

        # ── Panel 2: Charge sensor (dI/dVg) ────────────────────────────────
        ax = axes[1]
        sensor = result['charge_sensor']
        # Numerical derivative to show boundaries more clearly
        d_sensor = np.gradient(sensor, axis=0)**2 + np.gradient(sensor, axis=1)**2
        d_sensor = np.sqrt(d_sensor)
        im1 = ax.pcolormesh(Vg_x, Vg_y, d_sensor, cmap='hot_r',
                             shading='nearest')
        fig.colorbar(im1, ax=ax, label=r'$|{\nabla}I_{sensor}|$ (a.u.)')
        ax.set_xlabel(r'$V_{g1}$ (mV)')
        ax.set_ylabel(r'$V_{g2}$ (mV)')
        ax.set_title('Charge Sensor Signal')

        # ── Panel 3: Individual dot occupations ────────────────────────────
        ax = axes[2]
        combined = result['occupation_x'] + 2 * result['occupation_y']
        im2 = ax.pcolormesh(Vg_x, Vg_y, result['occupation_x'],
                             cmap='RdBu_r', shading='nearest',
                             vmin=0, vmax=result['occupation_x'].max())
        fig.colorbar(im2, ax=ax, label=r'$\langle N_1 \rangle$')
        ax.set_xlabel(r'$V_{g1}$ (mV)')
        ax.set_ylabel(r'$V_{g2}$ (mV)')
        ax.set_title(r'$\langle N_1 \rangle$ (Dot 1)')

        fig.suptitle(title, fontsize=14, fontweight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            for fmt in ('png', 'pdf'):
                fig.savefig(f"{save_path}.{fmt}", bbox_inches='tight')
                logger.info(f"Saved: {save_path}.{fmt}")
        plt.close(fig)
