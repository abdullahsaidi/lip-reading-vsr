import streamlit as st
import time
import base64
import tempfile
import os
from pathlib import Path

# Load the inference pipeline once at app start 
@st.cache_resource(show_spinner="Loading models (this happens once)...")
def load_pipeline():
    from inference import VSRPipeline
    pipeline = VSRPipeline(
        visual_ckpt=os.environ.get("VSR_VISUAL_CKPT", "models/avg_e4_e9.pth"),
        lm_path=os.environ.get("VSR_LM_PATH", "qwen3b_model"),
        lm_alpha=0.10,
        lm_beta=0.0,
    )
    pipeline.warmup()
    return pipeline


st.set_page_config(
    page_title="Lip Reading System",
    page_icon="👄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def find_background_image():
    files = list(Path(".").glob("psut_bg*"))
    if files:
        return files[0]
    return None

def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

bg_path = find_background_image()
bg_image = image_to_base64(bg_path) if bg_path else ""

if bg_image:
    hero_background = f"""
        linear-gradient(90deg,
            rgba(2, 6, 23, 0.99) 0%,
            rgba(2, 6, 23, 0.95) 34%,
            rgba(2, 6, 23, 0.82) 62%,
            rgba(2, 6, 23, 0.90) 100%
        ),
        url("data:image/jpeg;base64,{bg_image}")
    """
else:
    hero_background = """
        linear-gradient(135deg,
            rgba(15, 23, 42, 0.98),
            rgba(30, 41, 59, 0.72)
        )
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* {{ font-family: 'Inter', sans-serif; }}
html, body, [class*="css"] {{ color: #f8fafc; }}
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 12% 8%, rgba(59,130,246,0.16), transparent 26%),
        radial-gradient(circle at 88% 12%, rgba(168,85,247,0.14), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(34,211,238,0.08), transparent 32%),
        linear-gradient(135deg, #020617 0%, #071120 50%, #020617 100%);
}}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.block-container {{ max-width: 1180px; padding-top: 1.6rem; padding-bottom: 2rem; }}
.topbar {{ display: flex; justify-content: space-between; align-items: center; padding: 18px 24px; border-radius: 24px; background: rgba(15, 23, 42, 0.60); border: 1px solid rgba(148, 163, 184, 0.14); backdrop-filter: blur(14px); box-shadow: 0 20px 55px rgba(0,0,0,0.20); margin-bottom: 26px; }}
.brand {{ display: flex; align-items: center; gap: 13px; font-size: 21px; font-weight: 900; color: white; }}
.brand-icon {{ width: 44px; height: 44px; border-radius: 16px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #2563eb, #7c3aed); box-shadow: 0 14px 32px rgba(37, 99, 235, 0.32); }}
.top-pill {{ padding: 10px 18px; border-radius: 999px; background: rgba(37, 99, 235, 0.13); border: 1px solid rgba(96, 165, 250, 0.25); color: #bfdbfe; font-size: 13px; font-weight: 800; }}
.hero {{ position: relative; overflow: hidden; border-radius: 34px; min-height: 400px; padding: 62px 56px; background: {hero_background}; background-size: cover; background-position: center right; background-repeat: no-repeat; border: none; outline: none; box-shadow: 0 32px 90px rgba(0, 0, 0, 0.36); margin-bottom: 28px; }}
.hero::before {{ content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent 38%), radial-gradient(circle at 78% 42%, rgba(96,165,250,0.07), transparent 30%); pointer-events: none; }}
.hero::after {{ content: ""; position: absolute; inset: 0; border-radius: 34px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.035); pointer-events: none; }}
.hero-glow {{ position: absolute; width: 260px; height: 260px; right: -80px; top: -80px; background: rgba(96,165,250,0.12); filter: blur(80px); border-radius: 999px; }}
.hero-content {{ position: relative; z-index: 2; max-width: 650px; }}
.badge {{ display: inline-flex; align-items: center; gap: 8px; padding: 10px 17px; border-radius: 999px; background: rgba(59, 130, 246, 0.14); border: 1px solid rgba(96, 165, 250, 0.26); color: #93c5fd; font-size: 13px; font-weight: 900; margin-bottom: 20px; }}
.hero-title {{ font-size: 66px; line-height: 1.02; font-weight: 900; letter-spacing: -2.4px; margin: 0; color: #ffffff; }}
.gradient-text {{ background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.hero-text {{ margin-top: 22px; color: #dbe4f0; font-size: 18px; line-height: 1.8; max-width: 680px; }}
.hero-strip {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 30px; }}
.hero-chip {{ padding: 13px 16px; border-radius: 18px; background: rgba(2, 6, 23, 0.46); border: 1px solid rgba(148, 163, 184, 0.16); color: #e2e8f0; font-size: 13px; font-weight: 800; }}
.feature-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px; }}
.feature-card {{ padding: 24px; border-radius: 24px; background: rgba(15, 23, 42, 0.66); border: 1px solid rgba(148, 163, 184, 0.14); box-shadow: 0 20px 50px rgba(0,0,0,0.20); transition: 0.25s ease; }}
.feature-card:hover {{ transform: translateY(-3px); border-color: rgba(96,165,250,0.34); box-shadow: 0 26px 65px rgba(37,99,235,0.12); }}
.feature-title {{ color: #ffffff; font-size: 18px; font-weight: 900; margin-bottom: 8px; }}
.feature-text {{ color: #94a3b8; font-size: 14px; line-height: 1.7; }}
.panel-title {{ color: white; font-size: 29px; font-weight: 900; margin-bottom: 8px; }}
.panel-text {{ color: #94a3b8; font-size: 15px; line-height: 1.7; margin-bottom: 20px; }}
.info-card {{ padding: 20px; border-radius: 20px; background: rgba(30, 41, 59, 0.46); border: 1px solid rgba(148, 163, 184, 0.13); margin-bottom: 16px; }}
.waiting-card {{ padding: 22px; border-radius: 22px; background: rgba(250, 204, 21, 0.10); border: 1px solid rgba(250, 204, 21, 0.32); color: #fde68a; font-weight: 800; }}
.ready-card {{ padding: 22px; border-radius: 22px; background: rgba(74, 222, 128, 0.10); border: 1px solid rgba(74, 222, 128, 0.30); color: #bbf7d0; font-weight: 800; margin-bottom: 14px; }}
.processing-card {{ padding: 24px; border-radius: 22px; background: rgba(15, 23, 42, 0.78); border: 1px solid rgba(148, 163, 184, 0.16); margin-top: 14px; }}
.processing-label {{ color: #94a3b8; font-size: 12px; font-weight: 900; letter-spacing: 1.4px; text-transform: uppercase; margin-bottom: 10px; }}
.processing-stage {{ color: #ffffff; font-size: 22px; font-weight: 900; margin-bottom: 10px; }}
.processing-details {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }}
.detail-box {{ padding: 14px; border-radius: 16px; background: rgba(2, 6, 23, 0.52); border: 1px solid rgba(148, 163, 184, 0.14); }}
.detail-num {{ color: #ffffff; font-size: 22px; font-weight: 900; }}
.detail-label {{ color: #94a3b8; font-size: 12px; font-weight: 800; margin-top: 3px; }}
.result-card {{ padding: 28px; border-radius: 25px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(15, 23, 42, 0.86)); border: 1px solid rgba(74, 222, 128, 0.24); box-shadow: 0 26px 60px rgba(34, 197, 94, 0.10); margin-top: 18px; }}
.result-label {{ font-size: 13px; color: #86efac; font-weight: 900; letter-spacing: 1.2px; text-transform: uppercase; }}
.result-text {{ font-size: 50px; font-weight: 900; color: #ffffff; margin-top: 8px; letter-spacing: -1px; }}
.confidence-line {{ color: #dbe4f0; font-size: 15px; margin-top: 5px; }}
.lm-info {{ margin-top: 18px; padding: 16px; border-radius: 16px; background: rgba(2, 6, 23, 0.45); border: 1px solid rgba(148, 163, 184, 0.16); color: #cbd5e1; font-size: 14px; line-height: 1.7; }}
.stButton > button {{ width: 100%; padding: 16px 20px; border-radius: 18px; border: none; color: white; font-size: 16px; font-weight: 900; background: linear-gradient(90deg, #2563eb, #7c3aed, #0891b2); box-shadow: 0 18px 40px rgba(37, 99, 235, 0.30); transition: 0.25s ease; }}
.stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 22px 52px rgba(124, 58, 237, 0.40); }}
[data-testid="stFileUploader"] {{ padding: 18px; border-radius: 22px; background: rgba(30, 41, 59, 0.56); border: 1px dashed rgba(147, 197, 253, 0.32); }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 28px !important; background: rgba(15, 23, 42, 0.62) !important; border: 1px solid rgba(148, 163, 184, 0.15) !important; box-shadow: 0 22px 55px rgba(0,0,0,0.22) !important; backdrop-filter: blur(14px); }}
[data-testid="stMetric"] {{ padding: 15px; border-radius: 18px; background: rgba(2, 6, 23, 0.35); border: 1px solid rgba(148, 163, 184, 0.10); }}
.footer {{ text-align: center; color: #64748b; font-size: 13px; margin-top: 30px; }}
@media (max-width: 900px) {{
    .topbar {{ flex-direction: column; align-items: flex-start; gap: 12px; }}
    .hero {{ padding: 36px 28px; min-height: auto; background-position: center; }}
    .hero-title {{ font-size: 43px; }}
    .feature-grid {{ grid-template-columns: 1fr; }}
    .processing-details {{ grid-template-columns: 1fr; }}
}}
</style>
""", unsafe_allow_html=True)


def estimate_processing_time(uploaded_file):
    size_mb = uploaded_file.size / (1024 * 1024)
    estimated = 7.0 + (size_mb * 0.5)
    return round(min(max(estimated, 8.0), 25.0), 1)


def run_real_inference(uploaded_file):
    #Save uploaded file to temp, call the pipeline, return result dict.
    #Shows live progress during the inference time.

    # Save to temp file
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="Initializing video processing...")
    panel = st.empty()
    start_time = time.time()

    # Stage 1: preprocessing
    panel.markdown("""
    <div class="processing-card">
        <div class="processing-label">Live Processing Status</div>
        <div class="processing-stage">Detecting Mouth Region</div>
        <div style="color:#cbd5e1; line-height:1.7;">Running MediaPipe face detection and aligning mouth region...</div>
    </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(15, text="15% • Mouth region detection")

    pipeline = load_pipeline()

    try:
        video_tensor = pipeline.preprocess(tmp_path)
    except Exception as e:
        panel.empty()
        progress_bar.empty()
        os.unlink(tmp_path)
        raise e
    progress_bar.progress(45, text="45% • Visual model encoding")

    # Stage 2: visual model
    panel.markdown(f"""
    <div class="processing-card">
        <div class="processing-label">Live Processing Status</div>
        <div class="processing-stage">Running AI Prediction</div>
        <div style="color:#cbd5e1; line-height:1.7;">The visual speech model is analyzing {video_tensor.shape[0]} frames and generating top-50 hypotheses...</div>
    </div>
    """, unsafe_allow_html=True)
    hyps = pipeline.visual_inference(video_tensor)
    progress_bar.progress(70, text="70% • LM rescoring")

    # Stage 3: LM rescoring
    panel.markdown("""
    <div class="processing-card">
        <div class="processing-label">Live Processing Status</div>
        <div class="processing-stage">LLM Rescoring (Qwen2.5-3B)</div>
        <div style="color:#cbd5e1; line-height:1.7;">Reranking the 50 candidate transcriptions using a 3B-parameter language model...</div>
    </div>
    """, unsafe_allow_html=True)
    best_hyp, _ = pipeline.lm_rescore(hyps)
    progress_bar.progress(100, text="100% • Done")

    elapsed = time.time() - start_time
    os.unlink(tmp_path)

    panel.empty()
    progress_bar.empty()

    baseline_text = hyps[0]["text"] if hyps else ""
    final_text = best_hyp["text"] if best_hyp else baseline_text

    return {
        "text": final_text,
        "baseline_text": baseline_text,
        "rescued_by_lm": final_text != baseline_text,
        "rescued_rank": best_hyp["rank"] if best_hyp else 1,
        "n_hypotheses": len(hyps),
        "n_frames": video_tensor.shape[0],
        "total_time": elapsed,
    }


# UI 
st.markdown("""
<div class="topbar">
    <div class="brand">
        <div class="brand-icon">👄</div>
        <div>Lip Reading System</div>
    </div>
    <div class="top-pill">AI-Powered Visual Speech Recognition</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-glow"></div>
    <div class="hero-content">
        <div class="badge">✨ Princess Sumaya University for Technology</div>
        <h1 class="hero-title">
            Read lips.<br>
            Predict <span class="gradient-text">speech.</span>
        </h1>
        <div class="hero-text">
            An intelligent web-based system powered by a developed AI model to predict spoken words
            from silent videos through visual lip movement analysis and structured video processing.
        </div>
        <div class="hero-strip">
            <div class="hero-chip">01 Upload Video</div>
            <div class="hero-chip">02 Analyze Motion</div>
            <div class="hero-chip">03 Predict Text</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-title">🎥 Video Input</div>
        <div class="feature-text">Upload a short silent video and preview it directly inside the system.</div>
    </div>
    <div class="feature-card">
        <div class="feature-title">🧠 Developed AI Model</div>
        <div class="feature-text">Auto-AVSR Conformer fine-tuned on LRS2, with a Qwen2.5-3B LLM for rescoring.</div>
    </div>
    <div class="feature-card">
        <div class="feature-title">⚡ Live Processing</div>
        <div class="feature-text">A real-time processing panel shows stage, progress, elapsed time, and ETA.</div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large", border=True)

with left:
    st.markdown('<div class="panel-title">Upload Silent Video</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-text">Select a video file. For best results, keep the face clear, the mouth visible, and the clip short.</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded_video is not None:
        file_size_mb = uploaded_video.size / (1024 * 1024)
        estimated_time = estimate_processing_time(uploaded_video)

        st.markdown(f"""
        <div class="info-card">
            <div style="color:white; font-weight:900; font-size:16px;">Uploaded File</div>
            <div style="color:#94a3b8; margin-top:6px;">{uploaded_video.name}</div>
            <div style="color:#bfdbfe; margin-top:8px; font-weight:800;">
                Size: {file_size_mb:.2f} MB • Estimated processing time: {estimated_time:.1f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.video(uploaded_video)

with right:
    st.markdown('<div class="panel-title">Processing Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-text">Track the video processing status, estimated time, and final prediction result.</div>',
        unsafe_allow_html=True
    )

    if uploaded_video is None:
        st.markdown("""
        <div class="waiting-card">
            Waiting for video input. Upload a silent video to activate the processing console.
        </div>
        """, unsafe_allow_html=True)

    else:
        estimated_time = estimate_processing_time(uploaded_video)

        st.markdown(f"""
        <div class="ready-card">
            Video is ready for analysis. Expected processing time: {estimated_time:.1f} seconds.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Lip Reading Prediction"):
            try:
                result = run_real_inference(uploaded_video)

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Predicted Text</div>
                    <div class="result-text">"{result['text']}"</div>
                    <div class="confidence-line">
                        Processed in {result['total_time']:.1f}s • {result['n_frames']} frames • {result['n_hypotheses']} candidates evaluated
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show LM rescoring detail
                if result["rescued_by_lm"]:
                    st.markdown(f"""
                    <div class="lm-info">
                        <strong style="color: #93c5fd;">🧠 LM Rescoring Active</strong><br>
                        The Qwen2.5-3B language model improved the result.<br>
                        <strong>Visual model alone:</strong> "{result['baseline_text']}"<br>
                        <strong>After LM rescoring (rank {result['rescued_rank']}):</strong> "{result['text']}"
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="lm-info">
                        <strong style="color: #93c5fd;">🧠 LM Rescoring</strong><br>
                        The visual model's top prediction was already preferred by the LM.<br>
                        Final prediction = top of beam (rank 1).
                    </div>
                    """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Frames", result["n_frames"])
                m2.metric("Time", f"{result['total_time']:.1f}s")
                m3.metric("LM Used", "Qwen-3B")

            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.exception(e)

st.markdown("""
<div class="footer">
    Lip Reading System • Graduation Project • PSUT
</div>
""", unsafe_allow_html=True)
