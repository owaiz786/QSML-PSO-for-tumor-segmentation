# src/qmslpso_continuous.py
import numpy as np
from tqdm import tqdm

class QMSLPSO_Continuous:
    """
    A continuous version of the Quantum-behaved Multi-Swarm Learning Particle Swarm Optimizer.
    This optimizer is designed to find optimal hyperparameters within continuous bounds.
    """
    def __init__(self, fitness_evaluator, num_particles, param_bounds, num_swarms, generations, beta=1.0):
        """
        Initializes the optimizer.

        Args:
            fitness_evaluator (function): A function that takes a particle's position (a numpy array of parameters)
                                          and returns a single fitness score to be maximized.
            num_particles (int): The total number of particles in the population.
            param_bounds (dict): A dictionary defining the search space, e.g., {'lr': (1e-5, 1e-2), 'dropout': (0.1, 0.5)}.
            num_swarms (int): The number of sub-swarms to divide the population into.
            generations (int): The number of optimization iterations to run.
            beta (float): The Contraction-Expansion coefficient, controlling exploration/exploitation.
        """
        self.evaluator = fitness_evaluator
        self.num_particles = num_particles
        self.param_bounds = param_bounds
        self.param_keys = list(param_bounds.keys())
        self.num_dims = len(param_bounds)
        self.num_swarms = num_swarms
        self.generations = generations
        self.beta = beta

        # Extract min/max bounds for efficient clamping and initialization
        self.bounds_min = np.array([v for _, (v, _) in param_bounds.items()])
        self.bounds_max = np.array([v for _, (_, v) in param_bounds.items()])

        # --- Initialize particle states ---
        # 1. Positions (randomly within the defined bounds)
        self.positions = np.zeros((num_particles, self.num_dims))
        for i in range(self.num_dims):
            self.positions[:, i] = np.random.uniform(self.bounds_min[i], self.bounds_max[i], num_particles)
        
        # 2. Personal bests
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.full(num_particles, -np.inf) # Assuming maximization

        # 3. Swarm-level bests
        self.swarm_best_positions = np.zeros((num_swarms, self.num_dims))
        self.swarm_best_fitness = np.full(num_swarms, -np.inf)

        # 4. Global best
        self.global_best_position = np.zeros(self.num_dims)
        self.global_best_fitness = -np.inf
        
        # Divide particles into swarm groups
        self.swarm_indices = np.array_split(np.arange(num_particles), num_swarms)

        # History trackers for visualization
        self.convergence_history = []
        self.animation_history = []

    def _update_positions(self):
        """
        Calculates the new position for every particle based on the QPSO update rule.
        This is the core of the algorithm.
        """
        # Calculate the mean best position (mbest) across the entire population
        mbest = np.mean(self.pbest_positions, axis=0)

        for s_idx, indices in enumerate(self.swarm_indices):
            for p_idx in indices:
                # 1. Calculate the local attractor (p_i)
                # This version uses a simple average between personal best and the global best,
                # creating a strong pull towards the overall best solution found so far.
                p_i = (self.pbest_positions[p_idx] + self.global_best_position) / 2.0
                
                # 2. The core Quantum Update Rule
                u = np.random.rand(self.num_dims)
                
                # The sign is random, allowing for symmetric exploration around the attractor
                sign = np.where(np.random.rand(self.num_dims) > 0.5, 1, -1)
                
                self.positions[p_idx] = p_i + sign * self.beta * np.abs(mbest - self.positions[p_idx]) * np.log(1 / u)

        # 3. Enforce boundaries (clamping)
        # Ensure all new positions are within the valid hyperparameter ranges.
        self.positions = np.clip(self.positions, self.bounds_min, self.bounds_max)

    def optimize(self):
        """
        Runs the main optimization loop for a set number of generations.
        """
        for gen in tqdm(range(self.generations), desc="Optimizing Hyperparameters with QMSL-PSO"):
            
            # --- Record state for animation (optional, but good for analysis) ---
            # This captures the state at the beginning of the generation
            current_gen_swarm_ids = []
            for s_idx, indices in enumerate(self.swarm_indices):
                current_gen_swarm_ids.extend([s_idx] * len(indices))
            self.animation_history.append({
                'positions': np.copy(self.positions),
                'swarm_ids': np.array(current_gen_swarm_ids)
            })

            # --- 1. Evaluate Fitness ---
            for p_idx in range(self.num_particles):
                fitness = self.evaluator(self.positions[p_idx])
                
                # Update personal best if the new position is better
                if fitness > self.pbest_fitness[p_idx]:
                    self.pbest_fitness[p_idx] = fitness
                    self.pbest_positions[p_idx] = np.copy(self.positions[p_idx])

            # --- 2. Update Swarm and Global Bests ---
            current_global_best_updated = False
            for s_idx, indices in enumerate(self.swarm_indices):
                # Find the best particle within the current swarm
                best_particle_in_swarm_idx = indices[np.argmax(self.pbest_fitness[indices])]
                
                # Check if this swarm's best is better than its previous best
                if self.pbest_fitness[best_particle_in_swarm_idx] > self.swarm_best_fitness[s_idx]:
                    self.swarm_best_fitness[s_idx] = self.pbest_fitness[best_particle_in_swarm_idx]
                    self.swarm_best_positions[s_idx] = np.copy(self.pbest_positions[best_particle_in_swarm_idx])

            # Find the best among all swarms to update the global best
            best_swarm_idx = np.argmax(self.swarm_best_fitness)
            if self.swarm_best_fitness[best_swarm_idx] > self.global_best_fitness:
                self.global_best_fitness = self.swarm_best_fitness[best_swarm_idx]
                self.global_best_position = np.copy(self.swarm_best_positions[best_swarm_idx])
            
            self.convergence_history.append(self.global_best_fitness)

            # --- 3. Update Particle Positions for the Next Generation ---
            self._update_positions()

        print("\nQMSL-PSO Optimization Finished!")
        print(f"Best Fitness (Dice Score): {self.global_best_fitness:.4f}")
        
        # Convert the final best position vector into a readable dictionary
        best_params_dict = dict(zip(self.param_keys, self.global_best_position))
        print("Best Hyperparameters Found:", best_params_dict)
        
        return best_params_dict, self.convergence_history, self.animation_history