QMSL-PSO Optimized U-Net for Brain Tumor Segmentation

This project implements a state-of-the-art solution for automated brain tumor segmentation from MRI scans. It leverages a U-Net, a powerful deep learning architecture for biomedical imaging, and optimizes its hyperparameters using a novel metaheuristic algorithm: Quantum-behaved Multi-Swarm Learning Particle Swarm Optimization (QMSL-PSO).

The entire workflow is presented in a sophisticated and interactive Streamlit dashboard, which allows for live segmentation of new MRI scans, detailed performance comparisons, and in-depth analysis of the optimization process.

🌟 Key Features

U-Net for Segmentation: A deep learning model built from scratch in TensorFlow/Keras, specifically designed for precise pixel-level segmentation of medical images.

Advanced Hyperparameter Optimization: Uses a custom-built QMSL-PSO algorithm to automatically find the optimal learning_rate, dropout_rate, and number_of_filters for the U-Net.

Scientific Validation: The effectiveness of the QMSL-PSO algorithm is scientifically validated against a standard MSL-PSO on notoriously difficult benchmark functions (Rastrigin, Rosenbrock, Ackley).

Comprehensive Performance Analysis: The final, optimized model is rigorously compared against a baseline model using multiple metrics (Dice Score, F1-Score, IoU Score) and visual aids like confusion matrices.

Interactive Dashboard: A professional Streamlit application that provides:

A real-time tumor segmentation tool.

Detailed, interactive charts for model and algorithm comparison.

Visualizations of the optimization process, including an animated GIF of the swarm's exploration.

💻 Tech Stack

Backend & Deep Learning: Python 3.9+, TensorFlow / Keras

Optimization: Custom QMSL-PSO and MSL-PSO implementations in Python

Core Libraries: NumPy, Scikit-learn, Pillow (PIL), Scikit-image

Visualization: Streamlit, Matplotlib, Seaborn, Plotly, ImageIO

Data Handling: Pandas, Glob

📂 Project Structure
code
Code
download
content_copy
expand_less
QMSL-PSO-Tumor-Seg/
├── data/
│   └── archive/lgg-mri-segmentation/   # Kaggle dataset is unzipped here
├── reports/
│   ├── figures/                        # All PNG plots (comparison, matrices)
│   ├── benchmark_comparison_results.json # Results from benchmark tests
│   ├── best_hyperparameters.json       # Best params found for the U-Net
│   ├── comparison_results.json         # Final model performance metrics
│   ├── convergence_history.npy         # Data for convergence plots
│   ├── final_tumor_segmentation_model.h5 # The final, trained U-Net model
│   └── swarm_exploration.gif           # Animation of the optimizer
├── src/
│   ├── benchmark_functions.py          # Mathematical benchmark functions
│   ├── data_loader.py                  # Loads and preprocesses MRI data
│   ├── fitness.py                      # Fitness evaluator for the U-Net
│   ├── model.py                        # U-Net architecture and Dice metric
│   ├── mslpso_continuous.py            # Baseline (classical) PSO algorithm
│   ├── qmslpso_continuous.py           # The novel QMSL-PSO algorithm
│   └── viz.py                          # Generates all plots and animations
├── .venv/                              # Python virtual environment
├── dashboard.py                        # The main Streamlit dashboard application
├── run_benchmarks.py                   # Script to run the benchmark tests
├── run_optimizer.py                    # The main script to run the U-Net optimization
├── requirements.txt                    # All Python dependencies
└── README.md                           # This file
🚀 How to Run: A Step-by-Step Guide

Follow these instructions carefully to set up and run the entire project from start to finish. All commands should be run from the root directory of the project (QMSL-PSO-Tumor-Seg/).

Step 1: Setup and Installation

Clone the Repository

code
Bash
download
content_copy
expand_less
git clone <your-repository-url>
cd QMSL-PSO-Tumor-Seg

Create and Activate a Python Virtual Environment

code
Bash
download
content_copy
expand_less
# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate

Install Dependencies
This project has specific library requirements. Install them all with a single command:

code
Bash
download
content_copy
expand_less
pip install -r requirements.txt

Download the Dataset

Go to the LGG MRI Segmentation dataset on Kaggle.

Download the archive.zip file.

Unzip the file directly into the data/ folder. The final path should look like: data/archive/lgg-mri-segmentation/kaggle_3m/....

Step 2: Run the Main Optimization Pipeline

This is the core of the project where the QMSL-PSO finds the best hyperparameters and trains the final U-Net model.

What it does: This script will load the MRI data, run the QMSL-PSO for several generations (this is a long process), and then train a final model with the discovered hyperparameters. It will also train a baseline model for comparison and generate all the necessary result files and plots in the reports/ folder.

To run:

code
Bash
download
content_copy
expand_less
python run_optimizer.py

⚠️ Important: This step is computationally very expensive and can take several hours to complete, depending on your hardware (CPU/GPU). For your first test run, you may want to reduce the generations in run_optimizer.py from 5 to 2 to ensure the pipeline works.

Step 3: (Optional) Run the Scientific Benchmarks

This script validates the performance of your QMSL-PSO algorithm against a standard MSL-PSO on mathematical functions.

What it does: It runs both optimizers multiple times on the Rastrigin, Rosenbrock, and Ackley functions and saves the statistical results to a .json file.

To run:

code
Bash
download
content_copy
expand_less
python run_benchmarks.py
Step 4: Launch and Explore the Dashboard

Once Step 2 (and optionally Step 3) is complete, you can launch the interactive dashboard to see all the results.

What it does: It starts a local web server and opens the application in your browser. The dashboard will automatically load all the generated files from the reports/ directory.

To run:

code
Bash
download
content_copy
expand_less
streamlit run dashboard.py

You can now navigate through the different tabs to:

🔬 Live Segmentation: Upload your own MRI images and get instant tumor segmentations.

📊 Performance Comparison: See the clear, quantitative proof that the optimized model is superior to the baseline.

⚙️ Optimization Analytics: Dive deep into the optimization process with convergence plots and the swarm animation.

🧪 Algorithm Benchmarks: Analyze the scientific validation of the QMSL-PSO algorithm itself.
