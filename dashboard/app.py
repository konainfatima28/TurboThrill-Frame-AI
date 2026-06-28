import os
import sys
import yaml
import pandas as pd
import streamlit as st
from glob import glob

# Enforce script root execution compatibility context
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor.video_processor import VideoProcessor

# Page Layout Configuration
st.set_page_config(
    page_title="TurboThrill Spark Extractor AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Dark-Theme Injection via CSS
st.markdown("""
    <style>
        .main { background-color: #0f1116; color: #e2e8f0; }
        .stButton>button {
            background: linear-gradient(135deg, #ff4b4b 0%, #ff8533 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }
        .metric-card {
            background-color: #1a1f29;
            border: 1px solid #2d3748;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }
        .log-box {
            font-family: 'Courier New', Courier, monospace;
            background-color: #07090e !important;
            color: #39ff14 !important;
        }
        .gallery-card {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #2d3748;
            background-color: #1a1f29;
            padding: 5px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default_config.yaml")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_config():
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

# Initialize persistent session states using explicit keys
if "processed_videos_count" not in st.session_state:
    st.session_state["processed_videos_count"] = 0
if "total_sparks_extracted" not in st.session_state:
    st.session_state["total_sparks_extracted"] = 0
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def add_log(message: str):
    st.session_state["logs"].append(message)

# App Header Area
st.title("⚡ TurboThrill Spark Extractor AI")
st.caption("Production-Grade Computer Vision Pipeline for Kinetic Road Spark Detection")
st.markdown("---")

# Main Page Split Layout
left_panel, right_panel = st.columns([1, 2], gap="large")

with left_panel:
    st.subheader("🛠️ Pipeline Control Panel")
    config = load_config()

    # --- SECTION 1: INGESTION METHODOLOGY & CONTROLS ---
    st.markdown("### 📂 Ingestion & Execution Setup")
    input_mode = st.radio("Choose Video Ingestion Target:", ["Drag & Drop File Upload", "Local Directory Absolute Path Scanner"])
    
    target_files = []
    
    if input_mode == "Drag & Drop File Upload":
        uploaded_files = st.file_uploader(
            "Upload Video Stream Files", 
            type=["mp4", "mov", "avi", "mkv"], 
            accept_multiple_files=True,
            help="Drag one or multiple videos directly into the pipeline window."
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_path not in target_files:
                    target_files.append(file_path)
            st.success(f"Successfully buffered {len(target_files)} file(s) onto local extraction disk.")
            
    else:
        input_path = st.text_input("Source Video File / Folder Directory Absolute Path", value=os.getcwd())
        video_extensions = ('*.mp4', '*.mov', '*.avi', '*.mkv', '*.MP4', '*.MOV', '*.AVI', '*.MKV')
        if os.path.isdir(input_path):
            for ext in video_extensions:
                target_files.extend(glob(os.path.join(input_path, "**", ext), recursive=True))
        elif os.path.isfile(input_path):
            target_files.append(input_path)

    # Output selection parameter
    output_target = st.text_input("Extraction Output Directory Target", value=os.path.join(os.getcwd(), "outputs"))
    
    # Run button positioned precisely below the configuration fields
    st.markdown(" ")
    execute_pipeline = st.button("🚀 Run Extraction Pipeline", use_container_width=True, disabled=(len(target_files) == 0 and input_mode == "Drag & Drop File Upload"))
    st.markdown("---")

    # --- SECTION 2: TUNING PARAMETERS ---
    with st.expander("🎨 Computer Vision Hyperparameters", expanded=True):
        min_area = st.slider("Minimum Spark Area (pixels)", 1, 100, int(config["spark_detection"]["min_spark_area"]))
        max_area = st.slider("Maximum Spark Area (pixels)", 10, 300, 150)
        brightness_thresh = st.slider("Brightness (V-Channel) Threshold", 150, 255, int(config["spark_detection"]["brightness_threshold"]))
        sim_thresh = st.slider("Deduplication Perceptual Similarity", 0.50, 1.00, float(config["deduplication"]["similarity_threshold"]), 0.05)

    with st.expander("⚙️ System Processing Config", expanded=True):
        output_fmt = st.selectbox("Output Format Quality", ["JPG", "PNG"], index=0)
        frame_sampling = st.number_input("Frame Skip Interval (1 = check all)", min_value=1, value=int(config["processing"]["frame_sampling_rate"]))

    # Inject real-time overrides back into runtime config matrix
    config["spark_detection"]["min_spark_area"] = min_area
    config["spark_detection"]["max_spark_area"] = max_area
    config["spark_detection"]["brightness_threshold"] = brightness_thresh
    config["deduplication"]["similarity_threshold"] = sim_thresh
    config["processing"]["output_format"] = output_fmt.lower()
    config["processing"]["frame_sampling_rate"] = frame_sampling

with right_panel:
    st.subheader("📊 Engine Live Performance Workspace")
    
    # Hero Metric Matrix Grid (Reads session state directly)
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Total Streams Analyzed", value=st.session_state["processed_videos_count"], delta="Active batch runtime")
    with m_col2:
        st.metric(label="Validated Spark Frames Saved", value=st.session_state["total_sparks_extracted"])
    
    # Tab Component Routing for Clean Scannability
    tab_logs, tab_analytics, tab_gallery = st.tabs(["📋 System Output Logs", "📈 Data Telemetry Reports", "🖼️ Extracted Spark Gallery"])

    with tab_logs:
        if execute_pipeline:
            if not target_files:
                st.error("No valid video processing files detected in target queues. Adjust targets and retry.")
            else:
                st.session_state["logs"] = []
                add_log("[STARTUP] Initializing processing engines...")
                add_log(f"[INFO] Ingestion queue populated with {len(target_files)} target file streams.")
                
                processor = VideoProcessor(config)
                progress_bar = st.progress(0.0)
                sparks_found_run = 0
                videos_processed_run = 0
                
                for idx, video in enumerate(target_files):
                    add_log(f"[PROCESSING] Parsing -> {os.path.basename(video)}")
                    try:
                        extracted = processor.process_video(video, output_target, add_log)
                        sparks_found_run += extracted
                        videos_processed_run += 1
                    except Exception as e:
                        add_log(f"[ERROR] Engine encountered exception processing {os.path.basename(video)}: {str(e)}")
                    progress_bar.progress(float((idx + 1) / len(target_files)))
                    
                # Force synchronization update back to session memory structures
                st.session_state["processed_videos_count"] += videos_processed_run
                st.session_state["total_sparks_extracted"] += sparks_found_run
                st.success("🎉 Batch execution run finalized.")
                
                # Instantly re-render interface to push fresh values directly into metrics view
                st.rerun()
        
        st.text_area("Engine Core Feed", value="\n".join(st.session_state["logs"]), height=350, disabled=True)

    with tab_analytics:
        st.markdown("### 📊 Metadata Aggregation View")
        report_files = glob(os.path.join(output_target, "**", "report.csv"), recursive=True)
        if report_files:
            try:
                combined_df = pd.concat([pd.read_csv(f) for f in report_files], ignore_index=True)
                st.dataframe(combined_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### ⚡ Spark Confidence Distribution")
                st.bar_chart(data=combined_df, x="frame_number", y="spark_confidence", use_container_width=True)
            except Exception as e:
                st.info("Parsing performance report mappings...")
        else:
            st.info("No unified structural report records identified across current output directory targets.")

    with tab_gallery:
        st.markdown("### 🖼️ Extracted Spark Artifacts")
        if os.path.exists(output_target):
            discovered_images = []
            for ext in ('*.jpg', '*.png'):
                discovered_images.extend(glob(os.path.join(output_target, "**", ext), recursive=True))
                
            if discovered_images:
                sorted_imgs = sorted(discovered_images, reverse=True)[:24]
                st.markdown(f"Displaying up to **{len(sorted_imgs)}** latest high-fidelity spark events:")
                
                grid_cols = st.columns(3)
                for i, img_path in enumerate(sorted_imgs):
                    with grid_cols[i % 3]:
                        st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                        st.image(img_path, use_container_width=True)
                        st.caption(f"📍 {os.path.basename(img_path)}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No spark frames extracted yet. Launch the extraction pipeline to fill this gallery.")
        else:
            st.info("Waiting for pipeline to create the extraction folder structure.")