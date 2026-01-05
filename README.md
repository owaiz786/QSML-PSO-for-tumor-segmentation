# **QMSL-PSO Optimized U-Net for Brain Tumor Segmentation**

## **Overview**

This project presents a **high-performance, automated brain tumor segmentation framework** for MRI images, integrating **deep learning** with a **novel metaheuristic optimization strategy**. The core segmentation model is a **U-Net architecture**, specifically engineered for biomedical image analysis, whose critical hyperparameters are optimized using **Quantum-behaved Multi-Swarm Learning Particle Swarm Optimization (QMSL-PSO)**.

To ensure transparency, reproducibility, and usability, the complete workflow—from data preprocessing and model optimization to scientific benchmarking and inference—is deployed through a **professional, interactive Streamlit dashboard**. This enables real-time tumor segmentation, performance comparison, and in-depth visualization of the optimization dynamics.

---

## **Key Contributions**

### **1. Deep Learning–Based Medical Image Segmentation**

* Custom-built **U-Net architecture** implemented from scratch using **TensorFlow/Keras**.
* Designed for **pixel-level precision** in brain tumor segmentation tasks.
* Incorporates Dice-based loss and evaluation to address class imbalance in medical images.

### **2. Novel Hyperparameter Optimization Using QMSL-PSO**

* Introduces **Quantum-behaved Multi-Swarm Learning PSO**, a hybrid metaheuristic that improves:

  * Exploration–exploitation balance
  * Convergence stability
  * Avoidance of premature local minima
* Automatically optimizes:

  * Learning rate
  * Dropout rate
  * Number of convolutional filters

### **3. Scientific Benchmark Validation**

* QMSL-PSO is rigorously validated against **classical MSL-PSO** using challenging, non-convex benchmark functions:

  * **Rastrigin**
  * **Rosenbrock**
  * **Ackley**
* Statistical performance metrics and convergence behavior are recorded and analyzed.

### **4. Comprehensive Model Evaluation**

* Final optimized U-Net is quantitatively compared with a baseline U-Net using:

  * Dice Similarity Coefficient (DSC)
  * Intersection over Union (IoU)
  * F1-Score
* Includes:

  * Confusion matrices
  * Comparative plots
  * Visual segmentation outputs

### **5. Interactive Streamlit Dashboard**

A fully interactive dashboard enabling:

* **Live MRI tumor segmentation**
* **Side-by-side performance comparison**
* **Optimization convergence analytics**
* **Swarm exploration visualization (animated GIF)**
* **Benchmark algorithm analysis**

---

## **Technology Stack**

### **Core Technologies**

* **Python 3.9+**
* **TensorFlow / Keras**

### **Optimization & Algorithms**

* Custom **QMSL-PSO**
* Classical **MSL-PSO**

### **Data Processing & Evaluation**

* NumPy
* Scikit-learn
* Scikit-image
* Pillow (PIL)
* Pandas

### **Visualization & Dashboard**

* Streamlit
* Matplotlib
* Seaborn
* Plotly
* ImageIO

---

## **Project Structure**

```
QMSL-PSO-Tumor-Seg/
├── data/
│   └── archive/lgg-mri-segmentation/
│
├── reports/
│   ├── figures/
│   ├── benchmark_comparison_results.json
│   ├── best_hyperparameters.json
│   ├── comparison_results.json
│   ├── convergence_history.npy
│   ├── final_tumor_segmentation_model.h5
│   └── swarm_exploration.gif
│
├── src/
│   ├── benchmark_functions.py
│   ├── data_loader.py
│   ├── fitness.py
│   ├── model.py
│   ├── mslpso_continuous.py
│   ├── qmslpso_continuous.py
│   └── viz.py
│
├── dashboard.py
├── run_benchmarks.py
├── run_optimizer.py
├── requirements.txt
└── README.md
```

---

## **End-to-End Execution Guide**

### **Step 1: Environment Setup**

#### Clone Repository

```bash
git clone <your-repository-url>
cd QMSL-PSO-Tumor-Seg
```

#### Create Virtual Environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### **Step 2: Dataset Preparation**

1. Download the **LGG MRI Segmentation Dataset** from Kaggle.
2. Extract the dataset into:

```
data/archive/lgg-mri-segmentation/kaggle_3m/
```

---

### **Step 3: Run Optimization Pipeline**

This step performs:

* QMSL-PSO hyperparameter optimization
* Final U-Net training
* Baseline model training
* Metric evaluation and visualization generation

```bash
python run_optimizer.py
```

**Note:**
This process is computationally intensive and may take several hours. For testing purposes, reduce the number of generations in `run_optimizer.py`.

---

### **Step 4 (Optional): Run Benchmark Experiments**

Validates QMSL-PSO against MSL-PSO on mathematical benchmarks.

```bash
python run_benchmarks.py
```

---

### **Step 5: Launch Dashboard**

```bash
streamlit run dashboard.py
```

---

## **Dashboard Capabilities**

* **Live MRI Upload & Segmentation**
* **Optimized vs Baseline Model Comparison**
* **Convergence and Fitness Analysis**
* **Swarm Behavior Visualization**
* **Algorithm Benchmark Statistics**

---

## **Impact & Applications**

* Clinical decision support systems
* Automated radiology pipelines
* Medical AI research
* Optimization-driven deep learning architectures

---

