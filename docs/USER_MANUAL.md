# QuantumTwin: User Manual & API Guide

## 1. Quick Start Guide

### Installation
Ensure Python 3.10+ and Node.js 18+ are installed.
```bash
git clone https://github.com/tejapampana09/RESEARCH-PROJECT.git
cd QuantumTwin
pip install -r requirements.txt
```

### Running the Full Suite
To execute all 8 experiments, generate validation suites, and produce publication-ready figures:
```bash
python results/result_generator.py
python tests/validate_physics.py
python tests/benchmark.py
```

### Running the Interactive Web Dashboard
Start the FastAPI backend:
```bash
python backend/main.py
```
In another terminal, start the React Vite frontend:
```bash
cd frontend
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 2. API Reference

### Electrostatic Potential (`physics/potential.py`)
```python
from physics.potential import PotentialLandscape

pot = PotentialLandscape(grid_size=(45, 45), extent=(80e-9, 80e-9))
pot.add_dot("dot1", x0=0.0, y0=0.0, depth=80e-3, radius=12e-9)
pot.add_gate("P1", x0=25e-9, y0=0.0, lever_arm=0.12, sigma=15e-9)
V_map = pot.get_potential({"P1": 0.05})
```

### 2D Schrödinger Solver (`quantum/schrodinger.py`)
```python
from quantum.schrodinger import SchrodingerSolver2D

solver = SchrodingerSolver2D()
evals_eV, psi_list = solver.solve(V_map, pot.x, pot.y, num_states=4)
```

### Fermi-Hubbard Hamiltonian (`physics/hubbard.py`)
```python
from physics.hubbard import DoubleDotST

st = DoubleDotST(U1=4.0e-3, U2=4.0e-3, U12=0.8e-3, t0=50e-6)
res = st.compute_exchange(epsilon=0.0, V_b=0.0)
print(f"Exchange J = {res['J']*1e6:.2f} ueV")
```

### Charge Stability Honeycomb (`physics/charge_stability.py`)
```python
import numpy as np
from physics.charge_stability import ChargeStabilityDiagram

csd = ChargeStabilityDiagram(U_onsite=[4e-3, 4e-3], U_inter=[[0, 0.8e-3], [0.8e-3, 0]],
                             lever_arms=[[0.12, 0], [0, 0.12]], T=0.1)
result = csd.sweep(0, 1, np.linspace(0, 0.05, 40), np.linspace(0, 0.05, 40))
csd.plot(result, save_path="results/csd_output")
```

### RF Reflectometry (`physics/reflectometry.py`)
```python
from physics.reflectometry import RFReflectometrySimulator

sim = RFReflectometrySimulator(L_p=120e-9, C_p=1.0e-12, Q=80.0)
Gamma, amp, phase = sim.get_reflection(sim.f_0, C_q=10e-15)
I, Q, dtheta = sim.get_demodulated_iq(Gamma)
```

---

## 3. Running Unit Tests
```bash
python -m pytest tests/ -v
```
All 28 unit tests across core physics, noise models, reflectometry, Hubbard models, and charge stability will execute and confirm physical validity.
