# src/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import imageio
from tqdm import tqdm
import glob

# Ensure the output directory for figures exists
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_model_comparison(test_images, test_masks, baseline_model, optimized_model, num_samples=5):
    """
    Creates and saves a side-by-side comparison of segmentations.
    """
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(test_images))
        img = test_images[idx]
        true_mask = test_masks[idx]
        
        baseline_pred = (baseline_model.predict(np.expand_dims(img, axis=0))[0] > 0.5).astype(np.uint8)
        optimized_pred = (optimized_model.predict(np.expand_dims(img, axis=0))[0] > 0.5).astype(np.uint8)

        # Plot 1: Original Image + True Mask
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(true_mask[:,:,0], cmap='viridis', alpha=0.4)
        plt.title(f"Original + Ground Truth")
        plt.axis('off')

        # Plot 2: Baseline Model Prediction
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(baseline_pred[:,:,0], cmap='plasma', alpha=0.5)
        plt.title("Baseline Model Prediction")
        plt.axis('off')

        # Plot 3: QMSL-PSO Optimized Prediction
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(optimized_pred[:,:,0], cmap='plasma', alpha=0.5)
        plt.title("QMSL-PSO Optimized Prediction")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"))
    plt.close()
    print(f"\n[✔] Model comparison plot saved to {FIGURES_DIR}/model_comparison.png")


def create_swarm_animation_continuous(history, output_gif_path='reports/swarm_exploration.gif'):
    """
    Creates a GIF animating the swarm exploration in the 2D projected hyperparameter space.
    """
    print("\nGenerating swarm exploration animation...")
    all_positions = np.vstack([gen['positions'] for gen in history])
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_positions)
    
    FRAMES_DIR = os.path.join(FIGURES_DIR, "frames")
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    start_index = 0
    for i, gen_data in enumerate(tqdm(history, desc="Generating GIF frames")):
        num_particles = len(gen_data['positions'])
        end_index = start_index + num_particles
        
        gen_embedding = embedding[start_index:end_index]
        swarm_ids = gen_data['swarm_ids']
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=gen_embedding[:, 0], y=gen_embedding[:, 1], hue=swarm_ids, palette="viridis", s=50, alpha=0.8)
        plt.title(f'Hyperparameter Space Exploration - Generation {i + 1}')
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(title='Swarm ID')
        plt.xlim(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1)
        plt.ylim(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1)
        
        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        start_index = end_index

    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))
    with imageio.get_writer(output_gif_path, mode='I', fps=3) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"[✔] Animation saved to {output_gif_path}")

    for filename in frame_files: os.remove(filename)
    os.rmdir(FRAMES_DIR)