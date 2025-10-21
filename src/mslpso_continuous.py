# src/mslpso_continuous.py
import numpy as np
from tqdm import tqdm

class MSLPSO_Continuous:
    """
    A classical, continuous Multi-Swarm PSO to serve as a baseline for comparison.
    Uses the standard velocity and position update rules.
    """
    def __init__(self, fitness_evaluator, num_particles, param_bounds, num_swarms, generations, w=0.5, c1=1.5, c2=1.5, mode='max'):
        self.evaluator = fitness_evaluator
        self.num_particles = num_particles
        self.param_bounds = param_bounds
        self.param_keys = list(param_bounds.keys())
        self.num_dims = len(param_bounds)
        self.num_swarms = num_swarms
        self.generations = generations
        self.mode = mode
        
        # Classical PSO parameters
        self.w = w   # Inertia
        self.c1 = c1 # Cognitive coefficient
        self.c2 = c2 # Social coefficient

        self.bounds_min = np.array([v for _, (v, _) in param_bounds.items()])
        self.bounds_max = np.array([v for _, (_, v) in param_bounds.items()])

        # --- Initialize particle states ---
        self.positions = np.zeros((num_particles, self.num_dims))
        for i in range(self.num_dims):
            self.positions[:, i] = np.random.uniform(self.bounds_min[i], self.bounds_max[i], num_particles)
        
        # NEW: Initialize velocities
        self.velocities = np.random.randn(num_particles, self.num_dims) * 0.1
        
        self.pbest_positions = np.copy(self.positions)
        self.swarm_best_positions = np.zeros((num_swarms, self.num_dims))
        self.global_best_position = np.zeros(self.num_dims)
        
        if self.mode == 'max': initial_fitness = -np.inf
        else: initial_fitness = np.inf
        self.pbest_fitness = np.full(num_particles, initial_fitness)
        self.swarm_best_fitness = np.full(num_swarms, initial_fitness)
        self.global_best_fitness = initial_fitness
        
        self.swarm_indices = np.array_split(np.arange(num_particles), num_swarms)
        self.convergence_history = []
        self.animation_history = []  # Add this line

    def _update_velocity_and_position(self):
        """The core of the classical PSO."""
        for s_idx, indices in enumerate(self.swarm_indices):
            for p_idx in indices:
                r1, r2 = np.random.rand(self.num_dims), np.random.rand(self.num_dims)
                
                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[p_idx] - self.positions[p_idx])
                social_velocity = self.c2 * r2 * (self.swarm_best_positions[s_idx] - self.positions[p_idx])
                
                self.velocities[p_idx] = self.w * self.velocities[p_idx] + cognitive_velocity + social_velocity
                self.positions[p_idx] = self.positions[p_idx] + self.velocities[p_idx]

        self.positions = np.clip(self.positions, self.bounds_min, self.bounds_max)

    def optimize(self):
        desc = f"Optimizing ({self.mode}imization)"
        for gen in tqdm(range(self.generations), desc=desc):
            # Record state for animation (matching QMSLPSO format)
            current_gen_swarm_ids = []
            for s_idx, indices in enumerate(self.swarm_indices):
                current_gen_swarm_ids.extend([s_idx] * len(indices))
            self.animation_history.append({
                'positions': np.copy(self.positions),
                'swarm_ids': np.array(current_gen_swarm_ids)
            })
            
            for p_idx in range(self.num_particles):
                fitness = self.evaluator(self.positions[p_idx])
                is_better = fitness > self.pbest_fitness[p_idx] if self.mode == 'max' else fitness < self.pbest_fitness[p_idx]
                if is_better:
                    self.pbest_fitness[p_idx] = fitness
                    self.pbest_positions[p_idx] = np.copy(self.positions[p_idx])

            for s_idx, indices in enumerate(self.swarm_indices):
                best_local_idx = np.argmax(self.pbest_fitness[indices]) if self.mode == 'max' else np.argmin(self.pbest_fitness[indices])
                best_global_idx = indices[best_local_idx]
                is_swarm_better = self.pbest_fitness[best_global_idx] > self.swarm_best_fitness[s_idx] if self.mode == 'max' else self.pbest_fitness[best_global_idx] < self.swarm_best_fitness[s_idx]
                if is_swarm_better:
                    self.swarm_best_fitness[s_idx] = self.pbest_fitness[best_global_idx]
                    self.swarm_best_positions[s_idx] = np.copy(self.pbest_positions[best_global_idx])

            best_swarm_idx = np.argmax(self.swarm_best_fitness) if self.mode == 'max' else np.argmin(self.swarm_best_fitness)
            is_global_better = self.swarm_best_fitness[best_swarm_idx] > self.global_best_fitness if self.mode == 'max' else self.swarm_best_fitness[best_swarm_idx] < self.global_best_fitness
            if is_global_better:
                self.global_best_fitness = self.swarm_best_fitness[best_swarm_idx]
                self.global_best_position = np.copy(self.swarm_best_positions[best_swarm_idx])
            
            self.convergence_history.append(self.global_best_fitness)
            self._update_velocity_and_position()
        
        print(f"\nMSL-PSO Optimization Finished!")
        print(f"Best Fitness Found: {self.global_best_fitness:.6f}")
        
        best_params_dict = dict(zip(self.param_keys, self.global_best_position))
        print("Best Solution (Parameters) Found:", best_params_dict)
        
        return best_params_dict, self.convergence_history, self.animation_history