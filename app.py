import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px

# AgriVision Backend Modules
from src.config import REGIONAL_DEFAULTS, SUPPORTED_REGIONS
from src.analytics import FieldAnalytics
from src.inference import get_model, run_inference
from src.report import generate_pdf_report

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriVision Decision System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Custom Advanced CSS for Pixel-Perfect Layout
# ---------------------------------------------------------------------------
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global State Management
# ---------------------------------------------------------------------------
if "counts" not in st.session_state:
    st.session_state.counts = []
    st.session_state.annotated_img = None
    st.session_state.original_img = None


# ---------------------------------------------------------------------------
# Model Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Network...")
def load_agrivision_model():
    return get_model("iteration_2_tuned.pt")


# ---------------------------------------------------------------------------
# Top Hero Banner (Matches exact image setup)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">🌾</div>
    <div class="hero-text-container">
        <h1 class="hero-title">AgriVision Decision System</h1>
        <p class="hero-subtitle">High-Accuracy Wheat Head Detection Model (Iteration 2)</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Layout Geometry (Left = Visuals 65%, Right = Settings 35%)
# ---------------------------------------------------------------------------
col_main, col_settings = st.columns([1.8, 1])

# --- RIGHT COLUMN: ENGINE SETTINGS ---
with col_settings:
    # Header matching image
    st.markdown("""
    <div style="display:flex; align-items:center; border-bottom:1px solid #334155; padding-bottom:15px; margin-bottom:15px;">
        <span style="font-size:22px; margin-right:10px;">🧩</span>
        <span style="font-size:16px; font-weight:600; color:#F8FAFC; letter-spacing:1px;">ENGINE SETTINGS</span>
    </div>
    """, unsafe_allow_html=True)

    # Uploader
    uploaded_files = st.file_uploader(
        "Drag & drop or upload drone imagery (JPG, PNG)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    st.write("")
    
    # Sliders strictly styled
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.45, 0.05)
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.10, 0.90, 0.50, 0.05)
    
    st.write("")
    # Exact Architecture formatting from image
    st.markdown('<p class="arch-text">Architecture: YOLOv8s Multiscale</p>', unsafe_allow_html=True)
    st.markdown('<p class="arch-text" style="margin-bottom:15px;">Resolution: 1024px</p>', unsafe_allow_html=True)
    
    # --- COUNTRY/PARAM OVERRIDE (Hidden by default to keep clean UI) ---
    with st.expander("🌍 Agronomic Variables & Regional Setup", expanded=False):
        selected_region = st.selectbox(
            "Load Presets",
            options=SUPPORTED_REGIONS,
            index=SUPPORTED_REGIONS.index("Custom Setup")
        )
        reg_def = REGIONAL_DEFAULTS[selected_region]

        col_t, col_g = st.columns(2)
        with col_t:
            tgw = st.number_input("TGW (g)", value=float(reg_def["tgw_grams"]), step=1.0)
        with col_g:
            gph = st.number_input("Grains/Head", value=int(reg_def["grains_per_head"]), step=1)
            
        col_p, col_c = st.columns(2)
        with col_p:
            price = st.number_input("Price/Tonne", value=float(reg_def["price_per_tonne_usd"]), step=5.0)
        with col_c:
            currency = st.text_input("Currency", value=str(reg_def["currency"]))

    # --- THE GIANT STAT BOX (conditionally rendered) ---
    stat_placeholder = st.empty()


# --- INFERENCE TRIGGER ---
trigger_run = False

# --- LEFT COLUMN: STACKED VISUALS ---
with col_main:
    # 1. Original View
    st.markdown('<div class="panel-header">Original Drone Feed</div>', unsafe_allow_html=True)
    if st.session_state.original_img:
        st.image(st.session_state.original_img, use_container_width=True)
    else:
        # Placeholder styling
        st.markdown('<div style="height:200px; background-color:#1E293B; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#64748B;">Awaiting imagery...</div>', unsafe_allow_html=True)
    
    st.write("")
    # 2. Main execution button stacked between images
    if st.button("🚀 Run Biomass Analysis", type="primary", use_container_width=True):
        trigger_run = True
    st.write("")
    
    # 3. Processed View
    st.markdown('<div class="panel-header">Processed Network Output</div>', unsafe_allow_html=True)
    if st.session_state.annotated_img:
        st.image(st.session_state.annotated_img, use_container_width=True)
    else:
        st.markdown('<div style="height:200px; background-color:#1E293B; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#64748B;">Awaiting execution...</div>', unsafe_allow_html=True)


# --- EXECUTION LOOP ---
if trigger_run and uploaded_files:
    model = load_agrivision_model()
    counts = []
    
    with st.spinner("Executing spatial analysis..."):
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            st.session_state.original_img = img  # store last for display immediately
            
            count, ann_img = run_inference(
                model=model,
                image=img,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                img_size=1024
            )
            counts.append(count)
            st.session_state.annotated_img = ann_img
            
        st.session_state.counts = counts
        st.rerun()  # Forces a clean refresh so images immediately paint


# --- BOTTOM REPORTING & DASHBOARD ---
if st.session_state.counts:
    counts = st.session_state.counts
    engine = FieldAnalytics(counts)
    cv_pct = engine.calculate_cv()
    health = FieldAnalytics.get_health_status(cv_pct)
    mean_heads = float(np.mean(counts))
    
    yield_est = FieldAnalytics.estimate_yield(mean_heads, tgw=tgw, grains_per_head=gph)
    revenue = yield_est * price
    
    # Inject the Giant Box matching the screenshot EXACTLY
    with stat_placeholder.container():
        st.markdown(f"""
        <div class="stat-box-container">
            <p class="stat-number">{sum(counts):,}</p>
            <p class="stat-label">WHEAT HEADS DETECTED</p>
            <div class="stat-glow-line"></div>
            <div class="stat-success"><span>✅</span> Analysis Complete.</div>
        </div>
        """, unsafe_allow_html=True)

    # Financial & Export Row (Beneath the main two columns)
    st.markdown("<hr style='border:1px solid #334155; margin-top:30px; margin-bottom:30px;'>", unsafe_allow_html=True)
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Estimated Yield", f"{yield_est:.2f} t/ha")
    b_col2.metric("Spatial Uniformity (CV %)", f"{cv_pct:.1f}%", 
                  delta=health["message"], delta_color="inverse")
    b_col3.metric(f"Projected Revenue ({currency}/ha)", f"{revenue:,.2f} {currency}")
    
    with b_col4:
        st.markdown('<p class="panel-header" style="margin-bottom:5px;">Executive Briefing</p>', unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(
            counts=counts, mean_density=mean_heads, cv_pct=cv_pct, 
            yield_est=yield_est, revenue=revenue, health_msg=health["message"],
            currency=currency, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AgriVision_Analysis_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
