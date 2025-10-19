# dashboard.py
import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_title="QMSL-PSO Tumor Segmentation", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .hero-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    
    /* Upload zone */
    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e8ecf1 0%, #dce2e8 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Image container */
    .img-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        background: white;
        padding: 1rem;
    }
    
    .img-label {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 30px;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 1rem;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- File Paths ---
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODEL_PATH = os.path.join(REPORTS_DIR, 'final_tumor_segmentation_model.h5')
PARAMS_PATH = os.path.join(REPORTS_DIR, 'best_hyperparameters.json')
COMPARISON_RESULTS_PATH = os.path.join(REPORTS_DIR, 'comparison_results.json')
COMPARISON_IMG_PATH = os.path.join(FIGURES_DIR, "model_comparison.png")
ANIMATION_PATH = os.path.join(REPORTS_DIR, 'swarm_exploration.gif')
BASELINE_CM_PATH = os.path.join(FIGURES_DIR, "baseline_confusion_matrix.png")
OPTIMIZED_CM_PATH = os.path.join(FIGURES_DIR, "optimized_confusion_matrix.png")

# --- Helper Functions ---
@st.cache_resource
def load_keras_model(model_path):
    if os.path.exists(model_path):
        try:
            from src.model import dice_coef, dice_loss
            return tf.keras.models.load_model(
                model_path, 
                custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss}
            )
        except Exception as e:
            st.sidebar.error(f"Model load error: {str(e)[:100]}")
            return None
    return None

def preprocess_image(image_buffer):
    img = Image.open(image_buffer).convert('L').resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=[0, -1])

def create_overlay_image(original_img, mask):
    rgb_img = original_img.resize((128, 128)).convert('RGB')
    red_mask = np.zeros((128, 128, 4), dtype=np.uint8)
    red_mask[mask == 1] = [255, 0, 0, 150]
    overlay = Image.fromarray(red_mask, 'RGBA')
    rgb_img.paste(overlay, (0, 0), overlay)
    return rgb_img

def create_metrics_chart(baseline, optimized):
    """Create interactive comparison chart"""
    metrics = ['Dice Score', 'F1-Score', 'IoU Score']
    baseline_vals = [baseline['dice_score'], baseline['f1_score'], baseline['iou_score']]
    optimized_vals = [optimized['dice_score'], optimized['f1_score'], optimized['iou_score']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Baseline',
        x=metrics,
        y=baseline_vals,
        marker_color='#94a3b8',
        text=[f'{v:.4f}' for v in baseline_vals],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='QMSL-PSO Optimized',
        x=metrics,
        y=optimized_vals,
        marker_color='#667eea',
        text=[f'{v:.4f}' for v in optimized_vals],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Performance Comparison: Baseline vs Optimized',
        xaxis_title='Metric',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        barmode='group',
        template='plotly_white',
        height=400,
        font=dict(size=13),
        hovermode='x unified'
    )
    
    return fig

def calculate_tumor_stats(mask):
    """Calculate tumor statistics"""
    tumor_pixels = np.sum(mask)
    total_pixels = mask.size
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    return tumor_pixels, tumor_percentage

# --- Load Assets ---
model = load_keras_model(MODEL_PATH)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🧠 Navigation")
    st.markdown("---")
    
    # Model Status
    st.markdown("### 📊 Model Status")
    if model is not None:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Found")
    
    st.markdown("---")
    
    # Quick Stats
    if os.path.exists(COMPARISON_RESULTS_PATH):
        with open(COMPARISON_RESULTS_PATH, 'r') as f:
            results = json.load(f)
        
        st.markdown("### 📈 Quick Stats")
        improvement = results['optimized']['dice_score'] - results['baseline']['dice_score']
        st.metric("Improvement", f"+{improvement:.4f}")
        st.metric("Best Dice", f"{results['optimized']['dice_score']:.4f}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **QMSL-PSO** optimization for 
    brain tumor segmentation using 
    U-Net architecture.
    
    Upload MRI scans for instant 
    tumor detection and analysis.
    """)

# --- Main Header ---
st.markdown("""
<div class="hero-header">
    <h1>🧠 Brain Tumor Segmentation</h1>
    <p>Powered by QMSL-PSO Optimized U-Net</p>
</div>
""", unsafe_allow_html=True)

# --- Main Content ---
if not os.path.exists(REPORTS_DIR):
    st.error("⚠️ `reports` directory not found! Please run `python run_optimizer.py` first.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Live Segmentation",
        "📊 Performance Comparison",
        "📈 Optimization Analytics",
        "🔍 Detailed Analysis",
        "ℹ️ About"
    ])

    # --- Tab 1: Live Segmentation ---
    with tab1:
        st.markdown("## Real-Time Tumor Segmentation")
        
        if model is None:
            st.error("❌ Trained model not found. Please run `run_optimizer.py` first.")
        else:
            # Upload section
            st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
            st.markdown("### 📤 Upload Brain MRI Scan")
            uploaded_file = st.file_uploader(
                "Drag and drop or browse files",
                type=["tif", "png", "jpg", "jpeg"],
                help="Upload a grayscale brain MRI scan (TIF, PNG, JPG, JPEG)",
                label_visibility="collapsed"
            )
            st.markdown("**Supported formats:** TIF, PNG, JPG, JPEG • **Max size:** 200MB")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                # Display original image
                st.markdown("---")
                st.markdown("### 📋 Uploaded Scan")
                
                col_preview, col_info = st.columns([2, 1])
                
                original_image = Image.open(uploaded_file)
                with col_preview:
                    st.image(original_image, use_column_width=True)
                
                with col_info:
                    st.markdown("**Image Information**")
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                    st.write(f"**Type:** {uploaded_file.type}")
                    st.write(f"**Dimensions:** {original_image.size[0]}×{original_image.size[1]}")
                
                # Segment button
                st.markdown("---")
                if st.button("🔍 Segment Tumor", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("⚙️ Preprocessing image...")
                    progress_bar.progress(25)
                    
                    processed_image = preprocess_image(uploaded_file)
                    
                    status_text.text("🧠 Running model inference...")
                    progress_bar.progress(50)
                    
                    prediction = model.predict(processed_image, verbose=0)
                    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
                    
                    status_text.text("🎨 Creating visualizations...")
                    progress_bar.progress(75)
                    
                    # Calculate statistics
                    tumor_pixels, tumor_percentage = calculate_tumor_stats(mask)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Results
                    st.markdown("---")
                    st.markdown("### 🎯 Segmentation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="img-label">Original Scan</div>', unsafe_allow_html=True)
                        st.image(original_image, use_column_width=True)
                    
                    with col2:
                        st.markdown('<div class="img-label">Predicted Mask</div>', unsafe_allow_html=True)
                        st.image(mask * 255, use_column_width=True)
                    
                    with col3:
                        st.markdown('<div class="img-label">Tumor Overlay</div>', unsafe_allow_html=True)
                        st.image(create_overlay_image(original_image, mask), use_column_width=True)
                    
                    # Statistics
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Statistics")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    stat_col1.metric("Tumor Coverage", f"{tumor_percentage:.2f}%")
                    stat_col2.metric("Affected Pixels", f"{tumor_pixels:,}")
                    stat_col3.metric("Total Pixels", "16,384")
                    stat_col4.metric("Resolution", "128×128")
                    
                    st.markdown('<div class="success-box">✅ Segmentation completed successfully!</div>', unsafe_allow_html=True)

    # --- Tab 2: Performance Comparison ---
    with tab2:
        st.markdown("## Model Performance Comparison")
        
        if not os.path.exists(COMPARISON_RESULTS_PATH):
            st.warning("⚠️ Comparison results not found. Please run the updated `run_optimizer.py`.")
        else:
            with open(COMPARISON_RESULTS_PATH, 'r') as f:
                results = json.load(f)
            
            baseline_metrics = results['baseline']
            optimized_metrics = results['optimized']
            
            # Interactive chart
            st.plotly_chart(
                create_metrics_chart(baseline_metrics, optimized_metrics),
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### 📈 Key Performance Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📌 Baseline Model")
                st.metric("Dice Score", f"{baseline_metrics['dice_score']:.4f}")
                st.metric("F1-Score", f"{baseline_metrics['f1_score']:.4f}")
                st.metric("IoU (Jaccard)", f"{baseline_metrics['iou_score']:.4f}")
                st.markdown('<div class="info-box">Standard U-Net architecture without optimization</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🚀 QMSL-PSO Optimized Model")
                st.metric(
                    "Dice Score",
                    f"{optimized_metrics['dice_score']:.4f}",
                    delta=f"{optimized_metrics['dice_score'] - baseline_metrics['dice_score']:.4f}"
                )
                st.metric(
                    "F1-Score",
                    f"{optimized_metrics['f1_score']:.4f}",
                    delta=f"{optimized_metrics['f1_score'] - baseline_metrics['f1_score']:.4f}"
                )
                st.metric(
                    "IoU (Jaccard)",
                    f"{optimized_metrics['iou_score']:.4f}",
                    delta=f"{optimized_metrics['iou_score'] - baseline_metrics['iou_score']:.4f}"
                )
                st.markdown('<div class="success-box">Hyperparameters optimized using QMSL-PSO algorithm</div>', unsafe_allow_html=True)
            
            with st.expander("📖 Understanding These Metrics"):
                st.markdown("""
                **Dice Score (Sørensen–Dice Coefficient)**
                - Measures overlap between predicted and actual tumor regions
                - Range: 0 to 1 (higher is better)
                - Values > 0.7 are considered good for medical segmentation
                
                **F1-Score**
                - Harmonic mean of Precision and Recall
                - Balances false positives and false negatives
                - Critical for medical applications where both matter
                
                **IoU (Intersection over Union / Jaccard Index)**
                - Ratio of intersection to union of predicted and ground truth
                - Strict metric that penalizes both over and under-segmentation
                - Commonly used in computer vision benchmarks
                """)
            
            st.markdown("---")
            st.markdown("### 🎯 Pixel-Level Confusion Matrices")
            
            cm_col1, cm_col2 = st.columns(2)
            
            with cm_col1:
                st.markdown("#### Baseline Model")
                if os.path.exists(BASELINE_CM_PATH):
                    st.image(BASELINE_CM_PATH, use_column_width=True)
                else:
                    st.info("Baseline confusion matrix not found.")
            
            with cm_col2:
                st.markdown("#### Optimized Model")
                if os.path.exists(OPTIMIZED_CM_PATH):
                    st.image(OPTIMIZED_CM_PATH, use_column_width=True)
                else:
                    st.info("Optimized confusion matrix not found.")
            
            with st.expander("📖 Understanding Confusion Matrices"):
                st.markdown("""
                The confusion matrix shows pixel-level classification performance:
                
                - **True Negatives (TN)** - Top-left: Healthy pixels correctly classified
                - **False Positives (FP)** - Top-right: Healthy pixels misclassified as tumor
                - **False Negatives (FN)** - Bottom-left: Tumor pixels missed by model ⚠️
                - **True Positives (TP)** - Bottom-right: Tumor pixels correctly identified
                
                **In medical imaging, False Negatives are often most critical** - missing tumor 
                tissue can have serious consequences. The optimized model should show improvement 
                in reducing FN while maintaining good TP rates.
                """)
            
            st.markdown("---")
            st.markdown("### 🖼️ Visual Comparison on Test Images")
            
            if os.path.exists(COMPARISON_IMG_PATH):
                st.image(COMPARISON_IMG_PATH, use_column_width=True)
            else:
                st.info("Comparison visualization not found.")

    # --- Tab 3: Optimization Analytics ---
    with tab3:
        st.markdown("## Optimization Process Analysis")
        
        if not os.path.exists(PARAMS_PATH):
            st.warning("⚠️ Optimization results not found. Please run `run_optimizer.py`.")
        else:
            with open(PARAMS_PATH, 'r') as f:
                best_params = json.load(f)
            
            st.markdown("### 🏆 Best Hyperparameters Discovered")
            
            # Display parameters in a nice format
            param_cols = st.columns(3)
            param_items = list(best_params.items())
            
            for idx, (key, value) in enumerate(param_items):
                col_idx = idx % 3
                with param_cols[col_idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{key.replace('_', ' ').title()}</strong><br>
                        <span style="font-size: 1.5rem; color: #667eea;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("📊 View Raw JSON"):
                st.json(best_params)
            
            st.markdown("---")
            st.markdown("### 📈 Convergence Analysis")
            
            if os.path.exists('reports/convergence_history.npy'):
                data = np.load('reports/convergence_history.npy')
                
                # Create plotly figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode='lines+markers',
                    marker=dict(size=8, color='#667eea'),
                    line=dict(color='#764ba2', width=3),
                    name='Best Dice Score'
                ))
                
                fig.update_layout(
                    title='Optimization Convergence: Best Dice Score vs. Generation',
                    xaxis_title='Generation',
                    yaxis_title='Best Dice Score',
                    template='plotly_white',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                final_score = data[-1]
                initial_score = data[0]
                improvement = final_score - initial_score
                
                conv_col1, conv_col2, conv_col3, conv_col4 = st.columns(4)
                conv_col1.metric("Initial Score", f"{initial_score:.4f}")
                conv_col2.metric("Final Score", f"{final_score:.4f}")
                conv_col3.metric("Improvement", f"+{improvement:.4f}")
                conv_col4.metric("Generations", len(data))
            else:
                st.info("Convergence history not found.")
            
            st.markdown("---")
            st.markdown("### 🌊 Swarm Exploration Visualization")
            
            if os.path.exists(ANIMATION_PATH):
                st.markdown('<div class="info-box">This animation shows how the particle swarm explored the hyperparameter space to find optimal configurations.</div>', unsafe_allow_html=True)
                with open(ANIMATION_PATH, "rb") as f:
                    st.image(f.read(), use_column_width=True)
            else:
                st.info("Swarm animation GIF not found.")

    # --- Tab 4: Detailed Analysis ---
    with tab4:
        st.markdown("## Detailed Performance Analysis")
        
        if os.path.exists(COMPARISON_RESULTS_PATH):
            with open(COMPARISON_RESULTS_PATH, 'r') as f:
                results = json.load(f)
            
            # Calculate improvements
            baseline = results['baseline']
            optimized = results['optimized']
            
            dice_improvement = ((optimized['dice_score'] - baseline['dice_score']) / baseline['dice_score']) * 100
            f1_improvement = ((optimized['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
            iou_improvement = ((optimized['iou_score'] - baseline['iou_score']) / baseline['iou_score']) * 100
            
            st.markdown("### 📊 Percentage Improvements")
            
            imp_col1, imp_col2, imp_col3 = st.columns(3)
            
            imp_col1.metric("Dice Score Improvement", f"{dice_improvement:+.2f}%")
            imp_col2.metric("F1-Score Improvement", f"{f1_improvement:+.2f}%")
            imp_col3.metric("IoU Improvement", f"{iou_improvement:+.2f}%")
            
            st.markdown("---")
            st.markdown("### 📋 Complete Results JSON")
            st.json(results)
            
        else:
            st.warning("Detailed results not available.")

    # --- Tab 5: About ---
    with tab5:
        st.markdown("## About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This dashboard showcases an advanced brain tumor segmentation system that leverages:
        
        - **U-Net Architecture**: A powerful convolutional neural network designed for biomedical image segmentation
        - **QMSL-PSO Optimization**: Quantum-inspired Multi-Swarm Particle Swarm Optimization for hyperparameter tuning
        - **Real-time Processing**: Instant tumor detection and analysis from uploaded MRI scans
        
        ### 🔬 How It Works
        
        1. **Model Training**: A U-Net model is trained on brain MRI datasets
        2. **Hyperparameter Optimization**: QMSL-PSO searches for optimal model configurations
        3. **Comparison**: Performance is compared against a baseline model
        4. **Deployment**: The optimized model is deployed in this interactive dashboard
        
        ### 📈 Key Benefits
        
        - **Improved Accuracy**: QMSL-PSO optimization consistently improves segmentation quality
        - **Automated Tuning**: No manual hyperparameter selection required
        - **Visual Feedback**: Clear visualizations of tumor regions
        - **Quantitative Metrics**: Comprehensive performance evaluation
        
        ### 🛠️ Technology Stack
        
        - **Deep Learning**: TensorFlow / Keras
        - **Optimization**: Custom QMSL-PSO implementation
        - **Visualization**: Streamlit, Plotly, Matplotlib
        - **Image Processing**: PIL, NumPy
        
        ### 📊 Evaluation Metrics
        
        **Dice Score (Primary Metric)**
        - Measures spatial overlap between prediction and ground truth
        - Range: 0-1, where 1 is perfect segmentation
        - Formula: 2|A ∩ B| / (|A| + |B|)
        
        **F1-Score**
        - Balances precision and recall
        - Important for medical applications
        - Formula: 2 × (Precision × Recall) / (Precision + Recall)
        
        **IoU (Intersection over Union)**
        - Strict metric that penalizes both over- and under-segmentation
        - Formula: |A ∩ B| / |A ∪ B|
        
        ### 🚀 Getting Started
        
        1. Navigate to the **Live Segmentation** tab
        2. Upload a brain MRI scan (TIF, PNG, JPG)
        3. Click "Segment Tumor" to process
        4. View results and statistics
        5. Explore other tabs for detailed analysis
        
        ### 📝 Notes
        
        - Model processes 128×128 grayscale images
        - Red overlay indicates detected tumor regions
        - All metrics are computed on test dataset
        - Optimization history shows convergence over generations
        
        ### ⚠️ Disclaimer
        
        This tool is designed for research and educational purposes only. It should not be used 
        as a substitute for professional medical diagnosis or treatment. Always consult qualified 
        healthcare professionals for medical decisions.
        
        ### 📚 References & Resources
        
        - **U-Net Paper**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
        - **PSO**: Kennedy & Eberhart "Particle Swarm Optimization" (1995)
        - **Medical Imaging**: Brain tumor segmentation is a critical task in computer-aided diagnosis
        
        ### 🤝 Contributing
        
        To improve this project:
        1. Run `python run_optimizer.py` to train models with your own data
        2. Experiment with different QMSL-PSO parameters
        3. Compare results across multiple optimization runs
        4. Visualize and analyze the optimization trajectory
        
        ### 📧 Support
        
        For questions or issues:
        - Check that all required files are in the `reports/` directory
        - Ensure models are properly trained before launching dashboard
        - Review convergence plots to verify optimization quality
        
        ---
        
        **Built with ❤️ using TensorFlow, Streamlit, and QMSL-PSO**
        """)
        
        # Additional interactive elements
        st.markdown("---")
        st.markdown("### 🎓 Quick Tutorial")
        
        with st.expander("🔍 How to Use Live Segmentation"):
            st.markdown("""
            1. **Upload Image**: Click "Browse files" or drag-and-drop an MRI scan
            2. **Review Info**: Check image details in the information panel
            3. **Segment**: Click the "Segment Tumor" button
            4. **Analyze Results**: View the predicted mask and overlay
            5. **Check Statistics**: Review tumor coverage and pixel counts
            """)
        
        with st.expander("📊 How to Interpret Results"):
            st.markdown("""
            **Predicted Mask (White regions = Tumor)**
            - Shows binary classification of each pixel
            - White pixels indicate detected tumor tissue
            - Black pixels indicate healthy brain tissue
            
            **Tumor Overlay (Red overlay on original)**
            - Red transparent layer shows tumor location
            - Easier to visualize tumor in context
            - Helps assess anatomical position
            
            **Statistics**
            - **Tumor Coverage**: Percentage of image containing tumor
            - **Affected Pixels**: Total number of tumor pixels detected
            - **Resolution**: Image dimensions used for processing
            """)
        
        with st.expander("🧪 Understanding Optimization"):
            st.markdown("""
            **QMSL-PSO (Quantum-inspired Multi-Swarm Particle Swarm Optimization)**
            
            1. **Particle Swarm**: Population of candidate solutions exploring search space
            2. **Multi-Swarm**: Multiple sub-populations for better exploration
            3. **Quantum-inspired**: Uses quantum mechanics principles for enhanced search
            4. **Hyperparameters Optimized**:
               - Learning rate
               - Batch size
               - Number of filters
               - Dropout rates
               - And more...
            
            The algorithm iteratively improves solutions based on:
            - Personal best position of each particle
            - Global best position across all particles
            - Quantum-inspired exploration mechanisms
            """)
        
        st.markdown("---")
        
        # System requirements
        st.markdown("### 💻 System Requirements")
        
        req_col1, req_col2 = st.columns(2)
        
        with req_col1:
            st.markdown("""
            **Minimum Requirements:**
            - Python 3.8+
            - TensorFlow 2.x
            - 4GB RAM
            - CPU-based inference
            """)
        
        with req_col2:
            st.markdown("""
            **Recommended:**
            - Python 3.9+
            - TensorFlow 2.10+
            - 8GB+ RAM
            - GPU for training
            """)
        
        st.markdown("---")
        
        # File structure
        st.markdown("### 📁 Expected File Structure")
        
        st.code("""
reports/
├── final_tumor_segmentation_model.h5    # Trained model
├── best_hyperparameters.json            # Optimal hyperparameters
├── comparison_results.json              # Performance metrics
├── convergence_history.npy              # Optimization history
├── swarm_exploration.gif                # Animation
└── figures/
    ├── model_comparison.png             # Visual comparison
    ├── baseline_confusion_matrix.png    # Baseline CM
    └── optimized_confusion_matrix.png   # Optimized CM
        """, language="plaintext")
        
        st.markdown('<div class="info-box">💡 Ensure all files are present for full dashboard functionality</div>', unsafe_allow_html=True)