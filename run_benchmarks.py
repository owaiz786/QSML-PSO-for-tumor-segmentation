# run_benchmarks.py
import os
import json
import numpy as np
from tqdm import tqdm
from src.qmslpso_continuous import QMSLPSO_Continuous
from src.mslpso_continuous import MSLPSO_Continuous
from src.benchmark_functions import rastrigin, rosenbrock, ackley

os.makedirs("reports", exist_ok=True)
N_RUNS = 10 # Number of runs for statistical significance

def run_benchmark_comparison(func, bounds, name, dimensions=10, particles=50, generations=100):
    """Runs both QMSL-PSO and MSL-PSO and returns their comparative results."""
    print(f"\n--- Comparing Algorithms on: {name} Function ---")
    
    optimizers = {
        'QMSL-PSO': QMSLPSO_Continuous,
        'MSL-PSO': MSLPSO_Continuous
    }
    
    func_results = {}
    
    for opt_name, OptClass in optimizers.items():
        run_fitness_scores = []
        all_run_convergence = []
        
        for i in tqdm(range(N_RUNS), desc=f"Running {opt_name}"):
            param_bounds = {f'x{i}': bounds for i in range(dimensions)}
            optimizer = OptClass(
                fitness_evaluator=func, num_particles=particles, param_bounds=param_bounds,
                num_swarms=5, generations=generations, mode='min'
            )
            # Fixed: unpack 3 values instead of 2
            _, convergence, _ = optimizer.optimize()
            run_fitness_scores.append(optimizer.global_best_fitness)
            all_run_convergence.append(convergence)

        # Calculate statistics
        mean_fitness = np.mean(run_fitness_scores)
        std_fitness = np.std(run_fitness_scores)
        avg_convergence = np.mean(all_run_convergence, axis=0)
        
        func_results[opt_name] = {
            'mean_fitness': float(mean_fitness),
            'std_fitness': float(std_fitness),
            'convergence': [float(c) for c in avg_convergence]
        }
    return func_results

if __name__ == '__main__':
    all_results = {}
    all_results['rastrigin'] = run_benchmark_comparison(rastrigin, (-5.12, 5.12), "Rastrigin")
    all_results['rosenbrock'] = run_benchmark_comparison(rosenbrock, (-2.048, 2.048), "Rosenbrock")
    all_results['ackley'] = run_benchmark_comparison(ackley, (-32.768, 32.768), "Ackley")

    with open('reports/benchmark_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n[✔] All benchmark comparisons complete. Results saved to reports/benchmark_comparison_results.json")