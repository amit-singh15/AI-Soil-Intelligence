import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Soil Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS  – Premium Earth-Tone Light Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── App Background: warm sandy cream ── */
  .stApp {
      background: linear-gradient(150deg, #fdf6ec 0%, #f5ede0 40%, #eef5e8 100%);
      color: #2d1f0f;
  }

  /* ── Sidebar: deep warm brown ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #3b2006 0%, #5a3312 60%, #3b2006 100%) !important;
      border-right: 2px solid #c8853a;
  }
  [data-testid="stSidebar"] * { color: #f5ddb8 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #f5c06a !important; }
  [data-testid="stSidebar"] .stMarkdown p { color: #e8c99a !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(200,133,58,0.4) !important; }

  /* ── Hero banner ── */
  .hero {
      background: linear-gradient(135deg, rgba(101,163,13,0.14) 0%, rgba(202,138,4,0.10) 100%);
      border: 1.5px solid rgba(101,163,13,0.30);
      border-radius: 20px;
      padding: 2.2rem 2.5rem;
      margin-bottom: 1.8rem;
      display: flex;
      align-items: center;
      gap: 2rem;
      box-shadow: 0 4px 24px rgba(101,163,13,0.10);
  }
  .hero-logo {
      width: 110px; height: 110px;
      object-fit: contain; border-radius: 18px;
      box-shadow: 0 0 24px rgba(101,163,13,0.40), 0 0 50px rgba(101,163,13,0.12);
      flex-shrink: 0;
      border: 2px solid rgba(101,163,13,0.40);
      background: #fff;
      padding: 6px;
      animation: logoPulse 3s ease-in-out infinite;
  }
  @keyframes logoPulse {
      0%,100% { box-shadow: 0 0 24px rgba(101,163,13,0.40), 0 0 50px rgba(101,163,13,0.12); }
      50%      { box-shadow: 0 0 38px rgba(101,163,13,0.60), 0 0 70px rgba(101,163,13,0.22); }
  }
  .hero-text h1, .hero h1 {
      font-size: 2.4rem; font-weight: 800;
      background: linear-gradient(90deg, #5c7a1f, #84a729, #b97e2f);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin: 0 0 0.4rem 0;
  }
  .hero-text p, .hero p { color: #5a6e3a; font-size: 1.05rem; margin: 0; }

  /* ── SOIL IMAGE PANEL – bright white spotlight ── */
  [data-testid="stImage"] {
      background: #ffffff !important;
      border-radius: 18px !important;
      padding: 10px !important;
      box-shadow:
          0 0 0 3px #c8853a,
          0 0 0 6px rgba(200,133,58,0.25),
          0 12px 40px rgba(90,51,18,0.22) !important;
  }
  [data-testid="stImage"] img {
      border-radius: 12px !important;
      filter: contrast(1.08) saturate(1.12) brightness(1.04) !important;
  }
  [data-testid="stImage"] > div > small,
  [data-testid="stImage"] figcaption {
      color: #7a5230 !important;
      font-weight: 600 !important;
      font-size: 0.82rem !important;
      text-align: center !important;
      letter-spacing: 0.5px !important;
      margin-top: 0.5rem !important;
  }

  /* ── Metric cards ── */
  .metric-card {
      background: #ffffff;
      border: 1.5px solid rgba(101,163,13,0.25);
      border-radius: 16px;
      padding: 1.4rem;
      text-align: center;
      transition: transform 0.22s ease, box-shadow 0.22s ease;
      box-shadow: 0 2px 12px rgba(90,51,18,0.08);
      margin-bottom: 1rem;
  }
  .metric-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 10px 32px rgba(101,163,13,0.18), 0 2px 8px rgba(90,51,18,0.08);
      border-color: rgba(101,163,13,0.50);
  }
  .metric-label {
      font-size: 0.76rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 1.3px;
      color: #8a6440; margin-bottom: 0.6rem;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #3d7a1a; line-height: 1.1; }
  .metric-unit  { font-size: 0.85rem; color: #6a7e3f; font-weight: 500; }
  .metric-badge {
      display: inline-block; margin-top: 0.5rem;
      padding: 0.25rem 0.75rem; border-radius: 20px;
      font-size: 0.75rem; font-weight: 700;
  }

  /* ── Section titles ── */
  .section-title {
      font-size: 1.25rem; font-weight: 700; color: #4a6b1a;
      margin: 1.8rem 0 1rem 0;
      display: flex; align-items: center; gap: 0.6rem;
      border-bottom: 2px solid rgba(101,163,13,0.22);
      padding-bottom: 0.6rem;
  }

  /* ── Recommendation cards ── */
  .rec-card {
      background: #fff;
      border-left: 4px solid #84a729;
      border-radius: 0 14px 14px 0;
      padding: 1rem 1.3rem;
      margin-bottom: 0.8rem;
      box-shadow: 0 2px 10px rgba(90,51,18,0.07);
  }
  .rec-card h4 { color: #4a6b1a; margin: 0 0 0.3rem 0; font-size: 1rem; }
  .rec-card p  { color: #5c4a32; margin: 0; font-size: 0.9rem; line-height: 1.55; }

  /* ── Fertility badges ── */
  .fertility-high   { background: rgba(101,163,13,0.15); color: #3b6e10; border: 1.5px solid #65a30d; }
  .fertility-medium { background: rgba(202,138,4,0.15);  color: #92640a; border: 1.5px solid #ca8a04; }
  .fertility-low    { background: rgba(220,38,38,0.12);  color: #991b1b; border: 1.5px solid #dc2626; }

  /* ── Crop tags ── */
  .crop-tag {
      display: inline-block;
      background: rgba(101,163,13,0.10);
      border: 1.5px solid rgba(101,163,13,0.35);
      color: #4a6b1a;
      border-radius: 20px;
      padding: 0.3rem 0.9rem;
      font-size: 0.82rem; font-weight: 600;
      margin: 0.25rem;
  }

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] {
      border: 1.5px solid rgba(101,163,13,0.25) !important;
      border-radius: 12px !important;
      background: #fff !important;
  }

  /* ── Buttons ── */
  .stButton > button {
      background: linear-gradient(135deg, #5c7a1f, #84a729);
      color: #fff; border: none; border-radius: 12px;
      font-weight: 700; font-size: 1rem;
      padding: 0.7rem 2rem;
      transition: all 0.22s ease; width: 100%;
      box-shadow: 0 4px 14px rgba(84,107,25,0.30);
  }
  .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(84,107,25,0.40);
  }

  /* ── Progress bar ── */
  .stProgress > div > div {
      background: linear-gradient(90deg, #5c7a1f, #84a729, #c8853a) !important;
  }

  /* ── Alerts & captions ── */
  .stAlert { border-radius: 12px !important; }
  .stCaption, small { color: #7a5230 !important; }

  /* ── File uploader area ── */
  [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.06) !important;
      border: 2px dashed rgba(200,133,58,0.55) !important;
      border-radius: 14px !important;
  }
  [data-testid="stFileUploader"] section {
      background: rgba(0,0,0,0.25) !important;
      border-radius: 10px !important;
  }
  /* ── Camera input area ── */
  [data-testid="stCameraInput"] {
      background: rgba(255,255,255,0.06) !important;
      border: 2px dashed rgba(200,133,58,0.55) !important;
      border-radius: 14px !important;
  }
  [data-testid="stCameraInput"] > div > div {
      border-radius: 10px !important;
  }
  /* ── Input mode tab buttons ── */
  .img-mode-tabs { display:flex; gap:0.5rem; margin-bottom:0.8rem; }
  .img-mode-tab {
      flex:1; padding:0.55rem 0.3rem;
      border-radius: 10px;
      border: 1.5px solid rgba(200,133,58,0.40);
      background: rgba(0,0,0,0.20);
      color: #e8c99a;
      font-size: 0.82rem; font-weight:600;
      text-align:center; cursor:pointer;
      transition: all 0.2s ease;
      letter-spacing:0.3px;
  }
  .img-mode-tab.active {
      background: linear-gradient(135deg,rgba(200,133,58,0.35),rgba(245,192,106,0.20));
      border-color: #f5c06a;
      color: #f5c06a;
      box-shadow: 0 0 12px rgba(245,192,106,0.18);
  }
  .img-mode-tab:hover:not(.active) {
      background: rgba(200,133,58,0.15);
      border-color: rgba(200,133,58,0.70);
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
MODEL_PATH    = SCRIPT_DIR / "soil_ph_best_model.pkl"
DEFAULT_IMG   = SCRIPT_DIR / "Logo.png"
FEATURE_ORDER = ["R", "G", "B", "H", "S", "V"]

# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────
import base64 as _b64

def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as _f:
        return _b64.b64encode(_f.read()).decode()

if DEFAULT_IMG.exists():
    _logo_b64 = _img_to_b64(DEFAULT_IMG)
    st.markdown(f"""
<div class="hero">
  <img class="hero-logo" src="data:image/png;base64,{_logo_b64}" alt="AI Soil Analysis Logo" />
  <div class="hero-text">
    <h1>🌱 AI Soil Analysis Dashboard</h1>
    <p>Upload a soil image to instantly predict pH · N · P · K · Moisture · Soil Type ·
       Fertility Level · Crop Suitability · Fertilizer Recommendations — powered by Computer Vision &amp; ML.</p>
  </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="hero">
  <div class="hero-text">
    <h1>🌱 AI Soil Analysis Dashboard</h1>
    <p>Upload a soil image to instantly predict pH · N · P · K · Moisture · Soil Type ·
       Fertility Level · Crop Suitability · Fertilizer Recommendations — powered by Computer Vision &amp; ML.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────
def extract_color_features(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    mean_bgr = img_bgr.reshape(-1, 3).mean(axis=0)
    mean_rgb = mean_bgr[::-1]
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = img_hsv.reshape(-1, 3).mean(axis=0)
    return np.concatenate([mean_rgb, mean_hsv])

# ─────────────────────────────────────────────
# pH Classification
# ─────────────────────────────────────────────
def classify_ph(ph: float):
    if ph < 4.5:
        return "Extremely Acidic", "#ef4444"
    elif ph < 5.5:
        return "Strongly Acidic", "#f97316"
    elif ph < 6.5:
        return "Slightly Acidic", "#eab308"
    elif ph <= 7.0:
        return "Neutral (Ideal)", "#4ade80"
    elif ph <= 7.5:
        return "Slightly Alkaline", "#22d3ee"
    elif ph <= 8.5:
        return "Moderately Alkaline", "#818cf8"
    else:
        return "Strongly Alkaline", "#e879f9"

# ─────────────────────────────────────────────
# Derived Soil Parameters (heuristic models)
# ─────────────────────────────────────────────
def estimate_npk(ph: float, R: float, G: float, B: float) -> dict:
    brightness = (R + G + B) / 3.0
    N = max(10,  min(250, 180 - (brightness / 255.0) * 120 - abs(ph - 6.5) * 8))
    P = max(5,   min(150, 50  + (1 - abs(ph - 6.5) / 3.5) * 80 - (R - B) / 255.0 * 20))
    K = max(20,  min(280, 120 + (1 - brightness / 255.0) * 80 - max(0, ph - 8.0) * 15))
    return {"N": round(N, 1), "P": round(P, 1), "K": round(K, 1)}

def estimate_moisture(R: float, G: float, B: float, S: float) -> float:
    brightness = (R + G + B) / 3.0 / 255.0
    sat_norm   = S / 255.0
    moisture   = (1 - brightness) * 55 + sat_norm * 20 + 5
    return round(min(80, max(5, moisture)), 1)

def classify_soil_type(ph: float, R: float, G: float, B: float):
    brightness = (R + G + B) / 3.0
    redness    = R / (G + 1)
    if redness > 1.6 and ph < 6.5:
        return "Red Laterite / Ferralsol", "#ef4444"
    elif brightness < 80:
        return "Dark Clay / Vertisol", "#78716c"
    elif brightness > 200 and ph > 7.5:
        return "Sandy Loam / Aridisol", "#d97706"
    elif ph < 5.5:
        return "Acidic Forest / Spodosol", "#65a30d"
    elif 6.0 <= ph <= 7.5 and 100 < brightness < 180:
        return "Loamy / Inceptisol (Ideal)", "#22c55e"
    else:
        return "Silty Clay Loam / Mollisol", "#0ea5e9"

def fertility_level(N: float, P: float, K: float, ph: float):
    score  = min(N / 250 * 40, 40)
    score += min(P / 150 * 30, 30)
    score += min(K / 280 * 20, 20)
    score += max(0, 10 - abs(ph - 6.5) * 5)
    if score >= 70:
        return "High",   "fertility-high",   "🟢"
    elif score >= 40:
        return "Medium", "fertility-medium",  "🟡"
    else:
        return "Low",    "fertility-low",     "🔴"

def crop_suitability(ph: float, N: float, moisture: float) -> list:
    crops = [
        {"name": "🌾 Rice",       "ph_range": (5.0, 6.5), "n_min": 80,  "moist_min": 50},
        {"name": "🌽 Maize",      "ph_range": (5.5, 7.0), "n_min": 90,  "moist_min": 30},
        {"name": "🌿 Sugarcane",  "ph_range": (6.0, 7.5), "n_min": 100, "moist_min": 45},
        {"name": "🥔 Potato",     "ph_range": (4.8, 6.5), "n_min": 80,  "moist_min": 35},
        {"name": "🫘 Soybean",    "ph_range": (6.0, 7.0), "n_min": 40,  "moist_min": 30},
        {"name": "🌱 Wheat",      "ph_range": (6.0, 7.5), "n_min": 80,  "moist_min": 25},
        {"name": "☕ Coffee",      "ph_range": (4.5, 6.0), "n_min": 60,  "moist_min": 40},
        {"name": "🥜 Groundnut",  "ph_range": (5.5, 7.0), "n_min": 30,  "moist_min": 25},
        {"name": "🧅 Onion",      "ph_range": (6.0, 7.5), "n_min": 50,  "moist_min": 30},
        {"name": "🍅 Tomato",     "ph_range": (5.5, 7.0), "n_min": 70,  "moist_min": 35},
        {"name": "🥬 Spinach",    "ph_range": (6.5, 7.5), "n_min": 60,  "moist_min": 30},
        {"name": "🌻 Sunflower",  "ph_range": (6.0, 7.5), "n_min": 60,  "moist_min": 25},
    ]
    results = []
    for c in crops:
        score = (c["ph_range"][0] <= ph <= c["ph_range"][1]) * 50 + (N >= c["n_min"]) * 30 + (moisture >= c["moist_min"]) * 20
        if score >= 50:
            results.append({"crop": c["name"], "score": score})
    results.sort(key=lambda x: -x["score"])
    return results[:6]

def fertilizer_recommendations(N: float, P: float, K: float, ph: float) -> list:
    recs = []
    if ph < 6.0:
        recs.append({"title": "🪨 Lime Application (pH Correction)",
                     "detail": f"Soil pH ({ph}) is too acidic. Apply agricultural lime (CaCO₃) at 2–4 tonnes/hectare to raise pH closer to the ideal 6.5 and unlock nutrient availability."})
    elif ph > 8.0:
        recs.append({"title": "🧪 Sulphur / Acidifying Fertilizer",
                     "detail": f"Soil pH ({ph}) is too alkaline. Apply elemental sulphur at 200–500 kg/ha or use ammonium sulphate. Consider gypsum for sodic soils."})

    if N < 80:
        recs.append({"title": "🌿 Nitrogen Boost Required (N deficit)",
                     "detail": f"Estimated N = {N} mg/kg (target >120). Apply Urea (46-0-0) at 100–150 kg/ha or DAP. Consider legume cover crops for organic N fixation."})
    elif N > 200:
        recs.append({"title": "⚠️ Nitrogen Excess — Reduce N Inputs",
                     "detail": f"Estimated N = {N} mg/kg is high. Avoid additional nitrogen fertilizer this season to prevent nitrate runoff and crop lodging."})
    else:
        recs.append({"title": "✅ Nitrogen Level Adequate",
                     "detail": f"N = {N} mg/kg is within the optimal range. Maintain with a balanced organic compost top-dress (2–3 t/ha) mid-season."})

    if P < 30:
        recs.append({"title": "🔴 Phosphorus Deficiency Detected",
                     "detail": f"Estimated P = {P} mg/kg (target >50). Apply DAP (18-46-0) or SSP at 100–150 kg/ha. Rock phosphate is a slow-release organic option at 250 kg/ha."})
    elif P > 120:
        recs.append({"title": "ℹ️ Phosphorus Sufficient — Maintain",
                     "detail": f"P = {P} mg/kg is good. Skip additional P fertilization; excess P can cause zinc deficiency and environmental runoff."})

    if K < 80:
        recs.append({"title": "🟠 Potassium Supplement Needed",
                     "detail": f"Estimated K = {K} mg/kg (target >120). Apply MOP (Muriate of Potash, 0-0-60) at 80–120 kg/ha. For organic farms, use wood ash or banana peel compost."})

    if ph < 5.5 or ph > 8.0:
        recs.append({"title": "🔬 Micronutrient Check Advised",
                     "detail": "Extreme pH limits Zn, Fe, Mn, and B availability. Apply a chelated micronutrient foliar spray (ZnSO₄ 0.5% + Borax 0.2%) during early growth stages."})
    return recs

# ─────────────────────────────────────────────
# Full Analysis Pipeline
# ─────────────────────────────────────────────
def full_soil_analysis(image_path: str) -> dict:
    features        = extract_color_features(image_path)
    R, G, B, H, S, V = features
    features_df     = pd.DataFrame([features], columns=FEATURE_ORDER)
    raw_ph          = float(model.predict(features_df)[0])
    ph              = round(max(0.0, min(14.0, raw_ph)), 2)
    npk             = estimate_npk(ph, R, G, B)
    moisture        = estimate_moisture(R, G, B, S)
    soil_type, st_color        = classify_soil_type(ph, R, G, B)
    ph_class, ph_color         = classify_ph(ph)
    fert_level, fert_cls, fert_icon = fertility_level(npk["N"], npk["P"], npk["K"], ph)
    crops           = crop_suitability(ph, npk["N"], moisture)
    fertilizers     = fertilizer_recommendations(npk["N"], npk["P"], npk["K"], ph)
    return {
        "features":   dict(zip(FEATURE_ORDER, [round(x, 2) for x in features])),
        "ph": ph, "ph_class": ph_class, "ph_color": ph_color,
        "N": npk["N"], "P": npk["P"], "K": npk["K"],
        "moisture": moisture,
        "soil_type": soil_type, "st_color": st_color,
        "fert_level": fert_level, "fert_cls": fert_cls, "fert_icon": fert_icon,
        "crops": crops, "fertilizers": fertilizers,
    }

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Sidebar Logo ──
    if DEFAULT_IMG.exists():
        st.image(str(DEFAULT_IMG), use_container_width=True)
        st.markdown("""
        <div style="text-align:center;margin-top:-0.5rem;margin-bottom:1rem;">
          <span style="font-size:1.05rem;font-weight:700;color:#f5c06a;letter-spacing:1px;">AI Soil Analysis</span><br/>
          <span style="font-size:0.75rem;color:#e8c99a;">Powered by Computer Vision &amp; ML</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid rgba(200,133,58,0.35);'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.6rem;">
      <span style="font-size:1.3rem;">📷</span>
      <span style="font-size:1.1rem;font-weight:800;color:#f5c06a;letter-spacing:0.5px;">Image Input</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle ──
    input_mode = st.radio(
        "Input mode",
        options=["📁  Upload File", "📷  Use Webcam"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode_radio",
    )
    st.markdown(
        "<hr style='border:1px solid rgba(200,133,58,0.30);margin:0.5rem 0;'>",
        unsafe_allow_html=True,
    )

    uploaded_file = None
    webcam_image  = None

    if input_mode == "📁  Upload File":
        st.markdown(
            "<div style='font-size:0.78rem;color:#e8c99a;margin-bottom:0.4rem;'>📂 &nbsp;Select or drag a JPG/PNG soil image</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a soil image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Take a clear photo of bare soil under natural light.",
        )
        if uploaded_file:
            st.markdown(
                "<div style='font-size:0.75rem;color:#84cc6a;margin-top:0.3rem;'>✅ Image ready for analysis</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='font-size:0.78rem;color:#e8c99a;margin-bottom:0.4rem;'>📡 &nbsp;Point camera at bare soil &amp; capture</div>",
            unsafe_allow_html=True,
        )
        webcam_image = st.camera_input(
            "Capture soil image",
            label_visibility="collapsed",
            help="Point your camera at bare soil and click the capture button.",
        )
        if webcam_image:
            st.markdown(
                "<div style='font-size:0.75rem;color:#84cc6a;margin-top:0.3rem;'>✅ Photo captured — ready for analysis</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<hr style='border:1px solid rgba(200,133,58,0.30);margin:0.7rem 0 0.4rem 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown("### ℹ️ Tips for Best Results")
    st.markdown("""
- Use **natural daylight** — avoid flash
- Keep soil **moist but not flooded**
- Remove leaves/debris from frame
- Take from **~30 cm** above ground
- Cover a **uniform soil patch**
    """)
    st.markdown("---")
    analyse_btn = st.button("🔬 Run Full Soil Analysis", use_container_width=True)

# ─────────────────────────────────────────────
# Route: no image yet → landing page
# ─────────────────────────────────────────────
if uploaded_file is not None:
    tmp_path = SCRIPT_DIR / "temp_upload.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_path = str(tmp_path)
    should_run = True
elif webcam_image is not None:
    tmp_path = SCRIPT_DIR / "temp_upload.jpg"
    with open(tmp_path, "wb") as f:
        f.write(webcam_image.getbuffer())
    image_path = str(tmp_path)
    should_run = True
elif analyse_btn:
    image_path = str(DEFAULT_IMG)
    should_run = True
    st.info(f"ℹ️ No image provided — using default demo image: **{DEFAULT_IMG.name}**")
else:
    image_path = str(DEFAULT_IMG)
    should_run = False

if not should_run:
    # ── Animated CSS for landing page ──
    st.markdown("""
<style>
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(24px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes floatBadge {
  0%,100% { transform: translateY(0px);  }
  50%      { transform: translateY(-7px); }
}
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position:  400px 0; }
}
@keyframes countUp {
  from { opacity:0; transform:scale(0.7); }
  to   { opacity:1; transform:scale(1);   }
}
.step-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem; }
.step-card {
  background: linear-gradient(160deg, #3b2006 0%, #5a3312 60%, #3b2006 100%);
  border: 2px dashed #c8853a;
  border-radius: 14px;
  padding: 1.3rem 1.2rem 1.1rem 1.2rem;
  position: relative;
  overflow: hidden;
  animation: fadeInUp 0.6s ease both;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  cursor: default;
  box-shadow: 0 4px 18px rgba(59,32,6,0.45);
}
.step-card:hover { transform: translateY(-5px); box-shadow: 0 14px 38px rgba(200,133,58,0.30); border-color: #f5c06a; border-style: dashed; }
.step-card::before {
  content:'';
  position:absolute; inset:0;
  background: linear-gradient(90deg, transparent, rgba(200,133,58,0.07), transparent);
  background-size: 400px 100%;
  animation: shimmer 3s linear infinite;
}
.step-card:nth-child(1) { animation-delay:0.05s; }
.step-card:nth-child(2) { animation-delay:0.15s; }
.step-card:nth-child(3) { animation-delay:0.25s; }
.step-card:nth-child(4) { animation-delay:0.35s; }
.step-icon {
  font-size:2rem; margin-bottom:0.55rem; display:block;
  animation: floatBadge 3s ease-in-out infinite;
}
.step-card:nth-child(2) .step-icon { animation-delay:0.5s; }
.step-card:nth-child(3) .step-icon { animation-delay:1s;   }
.step-card:nth-child(4) .step-icon { animation-delay:1.5s; }
.step-num {
  position:absolute; top:0.8rem; right:1rem;
  font-size:3.2rem; font-weight:900; color:rgba(200,133,58,0.12);
  line-height:1; user-select:none;
}
.step-title { font-size:0.95rem; font-weight:700; color:#f5c06a; margin-bottom:0.35rem; letter-spacing:0.3px; }
.step-desc  { font-size:0.82rem; color:#e8c99a; line-height:1.55; }

.stats-strip {
  display:grid; grid-template-columns:repeat(4,1fr); gap:0.75rem;
  margin-top:1.4rem;
}
.stat-box {
  background: #fff;
  border: 1.5px solid rgba(200,133,58,0.30);
  border-radius:14px; padding:1rem 0.5rem;
  text-align:center;
  animation: countUp 0.7s ease both;
  box-shadow: 0 2px 10px rgba(90,51,18,0.07);
}
.stat-box:nth-child(1){animation-delay:0.4s}
.stat-box:nth-child(2){animation-delay:0.55s}
.stat-box:nth-child(3){animation-delay:0.7s}
.stat-box:nth-child(4){animation-delay:0.85s}
.stat-num  { font-size:1.55rem; font-weight:800; color:#5c7a1f; }
.stat-label{ font-size:0.7rem;  color:#8a6440; margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.8px; }

.pipeline-bar {
  display:flex; align-items:center; justify-content:space-between;
  background: rgba(255,255,255,0.80);
  border: 1.5px solid rgba(200,133,58,0.30);
  border-radius:50px; padding:0.65rem 1.4rem;
  margin-top:1.3rem; gap:0.4rem;
  box-shadow: 0 2px 10px rgba(90,51,18,0.07);
}
.pipe-step { color:#4a6b1a; font-size:0.78rem; font-weight:700; white-space:nowrap; }
.pipe-arrow { color:#c8853a; font-size:1rem; }
.pipe-arrow{ color:rgba(74,222,128,0.4); font-size:1rem; }
</style>

<div class="step-grid">
  <div class="step-card">
    <span class="step-num">1</span>
    <span class="step-icon">📸</span>
    <div class="step-title">Upload Soil Image</div>
    <div class="step-desc">Take a clear photo of bare soil under natural daylight — your smartphone camera is all you need.</div>
  </div>
  <div class="step-card">
    <span class="step-num">2</span>
    <span class="step-icon">🤖</span>
    <div class="step-title">AI pH Prediction</div>
    <div class="step-desc">Our ML model extracts R · G · B · H · S · V colour features and predicts soil pH in milliseconds.</div>
  </div>
  <div class="step-card">
    <span class="step-num">3</span>
    <span class="step-icon">🧪</span>
    <div class="step-title">N · P · K Analysis</div>
    <div class="step-desc">Heuristic models estimate Nitrogen, Phosphorus, Potassium, Moisture &amp; Soil Type from colour data.</div>
  </div>
  <div class="step-card">
    <span class="step-num">4</span>
    <span class="step-icon">🌾</span>
    <div class="step-title">Smart Recommendations</div>
    <div class="step-desc">Get personalised crop suitability rankings and targeted fertilizer action plans instantly.</div>
  </div>
</div>

<div class="stats-strip">
  <div class="stat-box"><div class="stat-num">12+</div><div class="stat-label">Crops Assessed</div></div>
  <div class="stat-box"><div class="stat-num">7</div><div class="stat-label">pH Classes</div></div>
  <div class="stat-box"><div class="stat-num">6</div><div class="stat-label">Soil Features</div></div>
  <div class="stat-box"><div class="stat-num">⚡</div><div class="stat-label">Real-time ML</div></div>
</div>

<div class="pipeline-bar">
  <span class="pipe-step">📷 Image</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step">🎨 RGB/HSV</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step">🤖 ML Model</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step">⚗️ pH Prediction</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step">🌿 N·P·K</span>
  <span class="pipe-arrow">→</span>
  <span class="pipe-step">✅ Report</span>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Run Analysis
# ─────────────────────────────────────────────
with st.spinner("🔬 Analysing soil… please wait"):
    try:
        res = full_soil_analysis(image_path)
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.stop()

# ─────────────────────────────────────────────
# ── Row 1: Image + Core Metrics ──
# ─────────────────────────────────────────────
col_img, col_metrics = st.columns([1, 2])

with col_img:
    st.image(image_path, caption="Analysed Soil Image", use_container_width=True)

with col_metrics:
    st.markdown('<div class="section-title">📊 Core Soil Parameters</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    with m1:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">pH Level</div>
  <div class="metric-value" style="color:{res['ph_color']}">{res['ph']}</div>
  <span class="metric-badge" style="background:rgba(74,222,128,0.15);color:{res['ph_color']};border:1px solid {res['ph_color']}">{res['ph_class']}</span>
</div>""", unsafe_allow_html=True)

    with m2:
        moist_label = '💧 Wet' if res['moisture'] > 55 else ('🌤 Moderate' if res['moisture'] > 30 else '🏜 Dry')
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Moisture</div>
  <div class="metric-value" style="color:#38bdf8">{res['moisture']}<span class="metric-unit"> %</span></div>
  <span class="metric-badge" style="background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid #38bdf8">{moist_label}</span>
</div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Fertility Level</div>
  <div class="metric-value" style="font-size:1.4rem">{res['fert_icon']} {res['fert_level']}</div>
  <span class="metric-badge {res['fert_cls']}">{res['fert_level']} Fertility</span>
</div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Nitrogen (N)</div>
  <div class="metric-value" style="color:#a78bfa">{res['N']}</div>
  <div class="metric-unit">mg / kg</div>
</div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Phosphorus (P)</div>
  <div class="metric-value" style="color:#fb923c">{res['P']}</div>
  <div class="metric-unit">mg / kg</div>
</div>""", unsafe_allow_html=True)

    with m6:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Potassium (K)</div>
  <div class="metric-value" style="color:#34d399">{res['K']}</div>
  <div class="metric-unit">mg / kg</div>
</div>""", unsafe_allow_html=True)

# ── pH Scale Indicator ──
st.markdown('<div class="section-title">🌡️ pH Scale Indicator</div>', unsafe_allow_html=True)
ph_col1, ph_col2 = st.columns([3, 1])
with ph_col1:
    st.progress(res['ph'] / 14.0)
    st.caption(f"pH {res['ph']} out of 14  ·  {res['ph_class']}")
with ph_col2:
    st.markdown("""<small style="color:#6b7280">
<b style="color:#ef4444">0–5</b> Very Acid &nbsp;|&nbsp;
<b style="color:#4ade80">6–7</b> Neutral &nbsp;|&nbsp;
<b style="color:#818cf8">8–14</b> Alkaline</small>""", unsafe_allow_html=True)

# ── Soil Type ──
st.markdown('<div class="section-title">🪨 Soil Type Classification</div>', unsafe_allow_html=True)

# Soil type descriptive highlights
soil_descriptions = {
    "Red Laterite / Ferralsol": "Rich in iron & aluminium oxides. Porous with good drainage, low organic matter, highly responsive to liming and organic manure.",
    "Dark Clay / Vertisol": "High moisture-holding capacity with deep cracking when dry. Nutrient-rich but requires careful water management.",
    "Sandy Loam / Aridisol": "Light and well-aerated with rapid drainage. Warms up quickly; benefits from regular organic mulching.",
    "Acidic Forest / Spodosol": "Coarse-textured and acidic. Benefit significantly from agricultural lime and balanced NPK amendments.",
    "Loamy / Inceptisol (Ideal)": "Well-balanced texture, excellent drainage and water retention. Highly fertile and suitable for a wide variety of crops.",
    "Silty Clay Loam / Mollisol": "Exceptionally fertile with high humus content and mineral richness. Superb moisture retention for grain & vegetable cultivation.",
}
soil_desc = soil_descriptions.get(res['soil_type'], "Identified based on extracted RGB chromatic profile and pH characteristics.")

r_val = int(res['features']['R'])
g_val = int(res['features']['G'])
b_val = int(res['features']['B'])

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #ffffff 0%, #faf6f0 100%);
    border: 1.5px solid rgba(200,133,58,0.35);
    border-left: 6px solid {res['st_color']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 20px rgba(90,51,18,0.08);
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease;
">
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 0.8rem;">
    <div style="display: flex; align-items: center; gap: 1rem;">
      <div style="
          width: 44px; height: 44px; border-radius: 12px;
          background: rgb({r_val}, {g_val}, {b_val});
          border: 3px solid #ffffff;
          box-shadow: 0 0 0 2px {res['st_color']}, 0 4px 10px rgba(0,0,0,0.15);
          flex-shrink: 0;
      " title="Extracted Soil Color: RGB({r_val}, {g_val}, {b_val})"></div>
      <div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #2d1f0f; line-height: 1.2;">{res['soil_type']}</div>
        <div style="font-size: 0.82rem; color: #7a5230; font-weight: 500; margin-top: 0.2rem;">Optical Classification via RGB &amp; HSV Spectrum</div>
      </div>
    </div>
    <span style="
        background: {res['st_color']}18;
        color: {res['st_color']};
        border: 1.5px solid {res['st_color']}55;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    ">
      ● {res['soil_type'].split('/')[0].strip()}
    </span>
  </div>

  <p style="color: #5c4a32; font-size: 0.9rem; line-height: 1.55; margin: 0.6rem 0 1rem 0; background: rgba(200,133,58,0.06); padding: 0.75rem 1rem; border-radius: 10px; border: 1px dashed rgba(200,133,58,0.25);">
    💡 <strong>Profile:</strong> {soil_desc}
  </p>

  <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;">
    <span style="background: #ffffff; border: 1px solid rgba(200,133,58,0.28); color: #6e4822; font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 8px;">
      🎨 <strong>RGB:</strong> {res['features']['R']} · {res['features']['G']} · {res['features']['B']}
    </span>
    <span style="background: #ffffff; border: 1px solid rgba(200,133,58,0.28); color: #6e4822; font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 8px;">
      🧪 <strong>pH:</strong> {res['ph']} ({res['ph_class']})
    </span>
    <span style="background: #ffffff; border: 1px solid rgba(200,133,58,0.28); color: #6e4822; font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.7rem; border-radius: 8px;">
      💧 <strong>Moisture Est:</strong> {res['moisture']}%
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── N P K Nutrient Chart ──
st.markdown('<div class="section-title">📈 Nutrient Profile (N · P · K)</div>', unsafe_allow_html=True)
npk_df = pd.DataFrame({"Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
                        "mg/kg":    [res["N"], res["P"], res["K"]]})
st.bar_chart(npk_df.set_index("Nutrient"), color="#22c55e", use_container_width=True)

# ── Colour Features ──
st.markdown('<div class="section-title">🎨 Extracted Colour Features</div>', unsafe_allow_html=True)
feat_df = pd.DataFrame([res["features"]])
st.dataframe(feat_df.style.format("{:.2f}").background_gradient(cmap="YlGn"), use_container_width=True)

# ── Crop Suitability ──
st.markdown('<div class="section-title">🌾 Crop Suitability</div>', unsafe_allow_html=True)
if res["crops"]:
    crop_html = "".join(f'<span class="crop-tag">{c["crop"]} &nbsp;·&nbsp; {c["score"]}%</span>' for c in res["crops"])
    st.markdown(crop_html, unsafe_allow_html=True)
    st.caption(f"Based on pH={res['ph']}, N={res['N']} mg/kg, Moisture={res['moisture']}%")
else:
    st.warning("⚠️ No well-suited crops found. Soil amendment strongly recommended before planting.")

# ── Fertilizer Recommendations ──
st.markdown('<div class="section-title">💊 Fertilizer & Amendment Recommendations</div>', unsafe_allow_html=True)
for rec in res["fertilizers"]:
    st.markdown(f'<div class="rec-card"><h4>{rec["title"]}</h4><p>{rec["detail"]}</p></div>', unsafe_allow_html=True)

# ── Full Summary Table ──
st.markdown('<div class="section-title">📋 Full Analysis Summary</div>', unsafe_allow_html=True)
summary = pd.DataFrame({
    "Parameter": ["pH", "pH Classification", "Nitrogen (N)", "Phosphorus (P)",
                  "Potassium (K)", "Moisture", "Soil Type", "Fertility Level"],
    "Value":     [res["ph"], res["ph_class"], f"{res['N']} mg/kg", f"{res['P']} mg/kg",
                  f"{res['K']} mg/kg", f"{res['moisture']} %", res["soil_type"], f"{res['fert_icon']} {res['fert_level']}"],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ──
st.markdown("""
<hr style="border:1px solid rgba(74,222,128,0.15);margin-top:2rem;">
<p style="text-align:center;color:#4b5563;font-size:0.82rem;">
  🌱 AI Soil Analysis — Colour-based ML predictions. Validate critical decisions with laboratory soil testing.
</p>""", unsafe_allow_html=True)