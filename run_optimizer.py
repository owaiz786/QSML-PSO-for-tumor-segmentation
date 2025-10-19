# run_optimizer.py
import os
import json
import numpy as np
import tensorflow as tf

from src.data_loader import load_data
from src.model import build_unet, dice_loss, dice_coef
from src.fitness import FitnessEvaluator
from src.qmslpso_continuous import QMSLPSO_Continuous
# ✅ --- Import new metric functions ---
from src.metrics import calculate_segmentation_metrics, plot_confusion_matrix
from src.viz import plot_model_comparison, create_swarm_animation_continuous

os.makedirs("reports/figures", exist_ok=True)

if __name__ == '__main__':
    # --- (Steps 1, 2, 3: Search Space, Optimization, Save Results are the same) ---
    param_bounds = {'learning_rate': (1e-5, 1e-2), 'dropout_rate': (0.1, 0.5), 'num_filters': (8, 32)}
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(sample_size=2000)
    
    evaluator = FitnessEvaluator()
    optimizer = QMSLPSO_Continuous(
        fitness_evaluator=evaluator.evaluate, num_particles=10, 
        param_bounds=param_bounds, num_swarms=2, generations=5
    ) # Your optimizer setup
    best_hyperparameters, convergence, animation_history = optimizer.optimize()

    serializable_params = {k: float(v) for k, v in best_hyperparameters.items()}
    with open('reports/best_hyperparameters.json', 'w') as f: json.dump(serializable_params, f, indent=4)
    np.save('reports/convergence_history.npy', np.array(convergence))
    
    # --- 4. Train Baseline and Optimized Models (same as before) ---
    print("\n--- Training Baseline Model ---")
    baseline_params = {'learning_rate': 0.001, 'dropout_rate': 0.3, 'num_filters': 16}
    baseline_model = build_unet((128, 128, 1), baseline_params['num_filters'], baseline_params['dropout_rate'])
    baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=baseline_params['learning_rate']), loss=dice_loss, metrics=[dice_coef])
    baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)
    
    # B) Final QMSL-PSO Optimized Model
    print("\n--- Training Final Optimized Model ---")
    optimized_model = build_unet((128, 128, 1), int(serializable_params['num_filters']), serializable_params['dropout_rate'])
    optimized_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=serializable_params['learning_rate']), loss=dice_loss, metrics=[dice_coef])
    optimized_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)
    # --- 5. ✅ UPGRADED: Evaluate Models with Detailed Metrics ---
    print("\n--- Evaluating Models on Test Set with Detailed Metrics ---")
    
    # Get predictions for the entire test set
    baseline_preds = (baseline_model.predict(X_test) > 0.5).astype(np.uint8)
    optimized_preds = (optimized_model.predict(X_test) > 0.5).astype(np.uint8)
    
    # Calculate the full suite of metrics
    baseline_metrics = calculate_segmentation_metrics(y_test, baseline_preds)
    optimized_metrics = calculate_segmentation_metrics(y_test, optimized_preds)
    
    print("\nBaseline Model Metrics:", json.dumps(baseline_metrics, indent=2))
    print("\nOptimized Model Metrics:", json.dumps(optimized_metrics, indent=2))
    
    # Save detailed comparison results
    comparison_results = {'baseline': baseline_metrics, 'optimized': optimized_metrics}
    with open('reports/comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # Save the final optimized model
    optimized_model.save('reports/final_tumor_segmentation_model.h5')
    
    # --- 6. ✅ UPGRADED: Generate All Visualizations including Confusion Matrices ---
    plot_model_comparison(X_test, y_test, baseline_model, optimized_model)
    create_swarm_animation_continuous(animation_history)
    
    # Generate and save the confusion matrix plots
    plot_confusion_matrix(baseline_metrics['raw_confusion_matrix'], 'Baseline')
    plot_confusion_matrix(optimized_metrics['raw_confusion_matrix'], 'Optimized')

    print("\nAll tasks complete. Dashboard is ready.")