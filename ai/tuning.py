"""
AI Gate Voltage Auto-Tuning and Reinforcement Learning Control
=============================================================

This module implements machine learning algorithms for autonomous tuning of quantum
dot devices. In particular, it contains a Deep Q-Network (DQN) Reinforcement Learning
agent that learns how to adjust plunger gate voltages (V_1, V_2) to tune a double
quantum dot array into the single-electron (1,1) regime.

Reinforcement Learning Formulation
----------------------------------
- State space: The current plunger gate voltages [V_1, V_2] (in Volts).
- Action space: 4 discrete actions:
  * Action 0: +dV to V_1 (Increase dot 1 plunger voltage)
  * Action 1: -dV to V_1 (Decrease dot 1 plunger voltage)
  * Action 2: +dV to V_2 (Increase dot 2 plunger voltage)
  * Action 3: -dV to V_2 (Decrease dot 2 plunger voltage)
- Reward function:
  * R = +10.0 if target state (e.g., (1, 1)) is reached. (Episode terminates)
  * R = +1.0 if intermediate states (1, 0) or (0, 1) are reached.
  * R = -1.0 if dots are completely empty (0, 0) to avoid getting stuck.
  * R = -0.5 for overcharged states (e.g., N_i > 2) to discourage leakage.
  * R = -0.05 step penalty to encourage rapid convergence.

Network Architecture
--------------------
A simple multilayer perceptron (MLP) mapping [V_1, V_2] to Q-values for the 4 actions:
    State (2) -> Dense (64, ReLU) -> Dense (64, ReLU) -> Q-Values (4)

References
----------
1. J. Darulova et al., "Autonomous tuning and stabilization of silicon spin qubits," Physical
   Review Applied 13, 054005 (2020).
2. R. Durrer et al., "Automated tuning of double quantum dots into the single-electron regime,"
   Physical Review Applied 13, 054019 (2020).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List, Dict, Any

class QNetwork(nn.Module):
    """
    Q-Network for approximating action-value function Q(s, a).
    """
    def __init__(self, state_dim: int = 2, action_dim: int = 4):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class DQNTuningAgent:
    """
    DQN Agent for autonomous quantum dot tuning.
    """
    def __init__(self, 
                 state_dim: int = 2, 
                 action_dim: int = 4, 
                 lr: float = 1e-3, 
                 gamma: float = 0.95,
                 epsilon_decay: float = 0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=2000)
        
        # Policy and Target Networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Stores experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32) -> float:
        """
        Trains the policy network on a random batch from memory.
        """
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor(np.array([float(x[4]) for x in batch])).unsqueeze(1)
        
        # Current Q-values
        curr_q = self.policy_net(states).gather(1, actions)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
            
        loss = self.loss_fn(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss.item())

    def update_target_network(self):
        """
        Copies policy network weights to target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def tune_device(self, 
                    device: Any, 
                    target_state: Tuple[int, int] = (1, 1), 
                    max_steps: int = 30,
                    dv: float = 3.0e-3) -> Dict[str, Any]:
        """
        Executes a live closed-loop auto-tuning run on a double dot device.
        Starts at current device voltages and steps plunger gate voltages to reach target_state.
        
        Args:
            device: A SiliconQDArray double-dot device instance.
            target_state: The target charge occupation (N1, N2).
            max_steps: Maximum tuning steps allowed.
            dv: Step voltage change in Volts (~0.3 meV of plunger shift).
            
        Returns:
            Dict containing the trajectory of gate voltages, charge states, and tuning status.
        """
        # Save original voltages
        v1_start = device.gate_voltages.get("P1", 0.0)
        v2_start = device.gate_voltages.get("P2", 0.0)
        
        v1, v2 = v1_start, v2_start
        
        trajectory_v = [[v1, v2]]
        trajectory_charge = []
        
        success = False
        step = 0
        
        # Run loop
        while step < max_steps:
            step += 1
            device.set_voltages({"P1": v1, "P2": v2})
            
            # Fetch current state (charge occupations)
            charge = device.get_charge_state(T_K=0.05).tolist()
            trajectory_charge.append(charge)
            
            # Verify if target reached
            rounded_charge = (int(round(charge[0])), int(round(charge[1])))
            if rounded_charge == target_state:
                success = True
                break
                
            # Form state vector for RL
            state_vec = np.array([v1, v2])
            
            # Select action (temporarily using greedy policy for tuning deployment)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values).item())
            
            # Apply action
            if action == 0:
                v1 += dv
            elif action == 1:
                v1 -= dv
            elif action == 2:
                v2 += dv
            elif action == 3:
                v2 -= dv
                
            # Clamp voltages within safety range [0, 0.2] Volts to prevent device damage
            v1 = np.clip(v1, 0.0, 0.15)
            v2 = np.clip(v2, 0.0, 0.15)
            
            trajectory_v.append([v1, v2])
            
        # Add final charge configuration
        device.set_voltages({"P1": v1, "P2": v2})
        final_charge = device.get_charge_state(T_K=0.05).tolist()
        trajectory_charge.append(final_charge)
        
        return {
            "success": success,
            "steps": step,
            "voltage_trajectory": trajectory_v,
            "charge_trajectory": trajectory_charge,
            "final_voltages": {"P1": v1, "P2": v2}
        }
        
    def pretrain_on_simulator(self, device: Any, target_state: Tuple[int, int] = (1, 1), episodes: int = 80):
        """
        Pre-trains the RL agent on the device simulator.
        
        This mimics running training in an offline simulator before deploying the policy on
        the real physical system.
        """
        dv = 4.0e-3  # Volts
        max_steps_per_episode = 25
        
        # Backup original voltages
        orig_voltages = device.gate_voltages.copy()
        
        for ep in range(episodes):
            # Random starting voltages in range [0.01, 0.08]
            v1 = random.uniform(0.01, 0.08)
            v2 = random.uniform(0.01, 0.08)
            device.set_voltages({"P1": v1, "P2": v2})
            
            state = np.array([v1, v2])
            
            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                
                # Take action
                next_v1, next_v2 = v1, v2
                if action == 0:
                    next_v1 += dv
                elif action == 1:
                    next_v1 -= dv
                elif action == 2:
                    next_v2 += dv
                elif action == 3:
                    next_v2 -= dv
                    
                next_v1 = np.clip(next_v1, 0.0, 0.12)
                next_v2 = np.clip(next_v2, 0.0, 0.12)
                
                # Apply next voltages
                device.set_voltages({"P1": next_v1, "P2": next_v2})
                
                # Get rewards
                charge = device.get_charge_state(T_K=0.05)
                n1, n2 = int(round(charge[0])), int(round(charge[1]))
                
                done = False
                if (n1, n2) == target_state:
                    reward = 10.0
                    done = True
                elif (n1, n2) == (0, 0):
                    reward = -1.0
                elif n1 > 2 or n2 > 2:
                    reward = -0.5
                elif n1 == 1 or n2 == 1:
                    reward = 1.0
                else:
                    reward = -0.05  # Step penalty
                    
                next_state = np.array([next_v1, next_v2])
                
                # Save transitions
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                v1, v2 = next_v1, next_v2
                
                if done:
                    break
                    
            # Perform training step
            self.replay(batch_size=16)
            
            # Periodically sync target network
            if ep % 5 == 0:
                self.update_target_network()
                
        # Restore original device voltages
        device.set_voltages(orig_voltages)
