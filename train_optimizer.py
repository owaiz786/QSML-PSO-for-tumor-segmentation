# train_optimizer.py
import json
from src.fitness import FitnessEvaluator
from src.qmslpso_continuous import QMSLPSO_Continuous
# ... more imports for final model training and viz

if __name__ == '__main__':
    # Define hyperparameter search space
    param_bounds = {
        'learning_rate': (1e-5, 1e-2),
        'dropout_rate': (0.1, 0.5),
        'num_filters': (8, 64) # e.g., 8, 16, 32, 64
    }

    evaluator = FitnessEvaluator()
    optimizer = QMSLPSO_Continuous(
        fitness_evaluator=evaluator.evaluate,
        num_particles=10, # Keep low due to long eval time
        param_bounds=param_bounds,
        num_swarms=2,
        generations=5 # Keep low for testing
    )

    best_hyperparameters = optimizer.optimize()

    # Save the best parameters found
    with open('reports/best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparameters, f)
    
    # --- Now, train a FINAL model with the best parameters for more epochs ---
    print("\nTraining final model with optimized parameters...")
    # ... (code to load full dataset, build model with best_params, train for 50 epochs, and save it) ...