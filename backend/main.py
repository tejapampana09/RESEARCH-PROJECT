"""
FastAPI Backend API for QuantumTwin Digital Twin
================================================

This module implements a FastAPI server that manages the state of the Digital Twin
and provides endpoints for visualization, control, and AI analytics.

Endpoints:
- GET /api/config: Get device configurations.
- POST /api/config: Update digital twin parameters (temperature, noise).
- GET /api/potential: Retrieve the current 2D potential landscape.
- POST /api/wavefunctions: Solve the Schrödinger equation and return energy states.
- POST /api/stability: Generate stability scan data (honeycomb pattern).
- POST /api/step: Run a single synchronized step of the twin with pink noise.
- POST /api/autotune: Trigger the RL autonomous tuning solver.
- GET /api/diagnose: Run device diagnostic checks.
- POST /api/predict_noise: Feed noise history to LSTM and get forecasts.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any, List, Optional
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.engine import DigitalTwinEngine
from ai.tuning import DQNTuningAgent
from ai.analytics import FaultDetector, NoisePredictor

app = FastAPI(title="QuantumTwin Backend API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances representing our active twin
twin_engine = DigitalTwinEngine(num_dots=2)  # Default: Double Quantum Dot
rl_agent = DQNTuningAgent(state_dim=2, action_dim=4)
fault_detector = FaultDetector()
noise_predictor = NoisePredictor()

# Warm up / pretrain the agent so it works on startup
print("Pretraining RL agent on simulator...")
rl_agent.pretrain_on_simulator(twin_engine.device, target_state=(1, 1), episodes=40)
print("RL agent ready.")

# Pydantic Schemas
class ConfigUpdate(BaseModel):
    temperature: Optional[float] = None
    noise_amplitude: Optional[float] = None
    noise_exponent: Optional[float] = None
    gate_voltages: Optional[Dict[str, float]] = None
    num_dots: Optional[int] = None

class WavefunctionRequest(BaseModel):
    num_states: int = 4

class StabilityRequest(BaseModel):
    v1_min: float = 0.0
    v1_max: float = 0.12
    v2_min: float = 0.0
    v2_max: float = 0.12
    steps: int = 60
    gate1: str = "P1"
    gate2: str = "P2"
    temperature: float = 0.05

class AutotuneRequest(BaseModel):
    target_state_x: int = 1
    target_state_y: int = 1
    max_steps: int = 25
    step_voltage: float = 3.5e-3

class NoisePredictRequest(BaseModel):
    history: List[float]

@app.get("/api/config")
def get_config():
    """
    Returns the current configuration parameters of the Digital Twin.
    """
    return {
        "num_dots": twin_engine.device.num_dots,
        "temperature": twin_engine.temperature_K,
        "noise_amplitude": twin_engine.noise_amplitude,
        "noise_exponent": twin_engine.noise_exponent,
        "gate_voltages": twin_engine.device.gate_voltages,
        "sensor_weights": twin_engine.device.sensor_weights.tolist(),
        "dots": [
            {"name": dot["name"], "x0": dot["x0"], "y0": dot["y0"], "radius": dot["radius"]}
            for dot in twin_engine.device.potential_landscape.dots
        ],
        "gates": [
            {"name": gate["name"], "x0": gate["x0"], "y0": gate["y0"], "type": gate["type"], "lever_arm": gate["lever_arm"]}
            for gate in twin_engine.device.potential_landscape.gates
        ]
    }

@app.post("/api/config")
def update_config(config: ConfigUpdate):
    """
    Updates the configuration of the digital twin.
    Re-creates the simulator if the number of dots changes.
    """
    global twin_engine, rl_agent
    
    if config.num_dots is not None and config.num_dots != twin_engine.device.num_dots:
        # Re-initialize device for different number of dots
        twin_engine = DigitalTwinEngine(num_dots=config.num_dots)
        # Re-initialize agent with appropriate dimensions (2 plungers for double, 3 for triple etc.)
        state_dim = config.num_dots
        # Simple action space: 2 * num_dots (increase/decrease for each plunger)
        action_dim = 2 * config.num_dots
        rl_agent = DQNTuningAgent(state_dim=state_dim, action_dim=action_dim)
        rl_agent.pretrain_on_simulator(twin_engine.device, target_state=tuple([1]*config.num_dots), episodes=20)
        
    update_data = {}
    if config.temperature is not None:
        update_data["temperature"] = config.temperature
    if config.noise_amplitude is not None:
        update_data["noise_amplitude"] = config.noise_amplitude
    if config.noise_exponent is not None:
        update_data["noise_exponent"] = config.noise_exponent
    if config.gate_voltages is not None:
        update_data["gate_voltages"] = config.gate_voltages
        
    twin_engine.update_twin_parameters(update_data)
    return get_config()

@app.get("/api/potential")
def get_potential():
    """
    Computes and returns the 2D potential energy map based on current voltages.
    """
    V = twin_engine.device.get_potential_map()
    x = twin_engine.device.potential_landscape.x
    y = twin_engine.device.potential_landscape.y
    
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "V": V.tolist()  # shape (Nx, Ny)
    }

@app.post("/api/wavefunctions")
def get_wavefunctions(req: WavefunctionRequest):
    """
    Solves the Schrödinger equation and returns the eigenvalues and wavefunction densities.
    """
    try:
        eigenvalues, wavefunctions = twin_engine.device.solve_quantum_states(num_states=req.num_states)
        
        # Prepare response
        states = []
        x = twin_engine.device.potential_landscape.x.tolist()
        y = twin_engine.device.potential_landscape.y.tolist()
        
        for idx in range(len(eigenvalues)):
            psi = wavefunctions[idx]
            density = (np.abs(psi) ** 2).tolist()  # Probability density
            states.append({
                "energy_eV": float(eigenvalues[idx]),
                "density": density
            })
            
        return {
            "x": x,
            "y": y,
            "states": states
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schrödinger Solver failed: {str(e)}")

@app.post("/api/stability")
def get_stability(req: StabilityRequest):
    """
    Performs a 2D gate voltage sweep to generate the charge stability diagram.
    """
    v1_range = np.linspace(req.v1_min, req.v1_max, req.steps)
    v2_range = np.linspace(req.v2_min, req.v2_max, req.steps)
    
    scan = twin_engine.device.get_stability_diagram(
        v1_range=v1_range,
        v2_range=v2_range,
        gate1=req.gate1,
        gate2=req.gate2,
        T_K=req.temperature,
        max_electrons=3
    )
    
    # Format stability diagram for frontend plotting
    return {
        "v1_range": v1_range.tolist(),
        "v2_range": v2_range.tolist(),
        "sensor": scan["sensor"].tolist(),
        "charge1": scan["charge1"].tolist(),
        "charge2": scan["charge2"].tolist(),
        "derivative": scan["derivative"].tolist()
    }

@app.post("/api/step")
def run_step():
    """
    Runs a single real-time synchronization step, adding 1/f noise to voltages.
    """
    state = twin_engine.get_synchronized_state()
    return state

@app.post("/api/autotune")
def trigger_autotune(req: AutotuneRequest):
    """
    Instructs the RL agent to autonomously adjust gate voltages
    to reach the desired charge state.
    """
    if twin_engine.device.num_dots != 2:
        raise HTTPException(status_code=400, detail="RL Autotuning currently only supported for Double Quantum Dot array.")
        
    target = (req.target_state_x, req.target_state_y)
    
    result = rl_agent.tune_device(
        device=twin_engine.device,
        target_state=target,
        max_steps=req.max_steps,
        dv=req.step_voltage
    )
    
    # Synchronize the twin engine's internal voltages to the final voltages found
    twin_engine.update_twin_parameters({"gate_voltages": result["final_voltages"]})
    
    return result

@app.get("/api/diagnose")
def run_diagnose():
    """
    Runs fault detection based on device parameters and noise level.
    """
    alpha = twin_engine.device.ci_model.alpha
    noise_amp = twin_engine.noise_amplitude
    num_dots = twin_engine.device.num_dots
    
    diagnosis = fault_detector.diagnose(
        lever_arm_matrix=alpha,
        nominal_noise_amp=noise_amp,
        num_dots=num_dots
    )
    return diagnosis

@app.post("/api/predict_noise")
def predict_noise(req: NoisePredictRequest):
    """
    Trains on past noise logs and predicts the next noise value.
    """
    try:
        # Dynamically fit the LSTM on the provided history
        loss = noise_predictor.train_on_history(req.history, epochs=3)
        prediction = noise_predictor.predict_next(req.history)
        
        return {
            "prediction": prediction,
            "training_loss": loss
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise predictor failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
