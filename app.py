"""
app.py  (v2 - improved interface + basic non-food image check)
------------------------------------------------------------------
Run:  streamlit run app.py

Only this file changed. food_model.h5, train.py, predict.py and calories.py
stay exactly the same - NO need to train again.
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from calories import DISCLAIMER, estimate_calories, pretty_name
from predict import load_food_model, predict_food

st.set_page_config(page_title="Food Calorie Estimator", page_icon="🍽️", layout="wide")

# ---------------- Styling ----------------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {
    background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
    padding: 28px 34px; border-radius: 18px; margin-bottom: 26px;
}
.hero h1 {margin: 0; font-size: 2rem; color: #ffffff !important; padding: 0;}
.hero p  {margin: 8px 0 0 0; color: #fff7f0; font-size: 1.02rem;}
.card {
    border: 1px solid rgba(128,128,128,0.25); border-radius: 16px;
    padding: 22px 24px; margin-bottom: 14px;
}
.label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.65;}
.food-name {font-size: 2.1rem; font-weight: 700; margin: 2px 0 10px 0;}
.kcal {font-size: 3.2rem; font-weight: 800; color: #ff6b3d; line-height: 1.1;}
.kcal small {font-size: 1.1rem; font-weight: 500; opacity: 0.8;}
.chip {
    display: inline-block; padding: 4px 12px; margin: 6px 6px 0 0;
    border-radius: 999px; background: rgba(255,107,61,0.13); font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🍽️ Food Calorie Detection and Estimation Using CNN</h1>
  <p>Upload a food photo - the CNN identifies the food and estimates its approximate calories.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load model ----------------
try:
    load_food_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ---------------- Helper: is this a real photo? ----------------
def photo_check(img_rgb):
    """
    Very simple check to reject screenshots / documents / drawings.
    Real food photos have texture everywhere, screenshots have large flat areas
    (plain white or single-colour). This is a heuristic, not perfect.
    Returns (looks_like_photo, flat_ratio, white_ratio)
    """
    small = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    flat_ratio = float((magnitude < 2).mean())            # pixels with no change at all
    white_ratio = float((small.min(axis=2) > 235).mean())  # near-white pixels
    ok = not (flat_ratio > 0.5 or white_ratio > 0.6)
    return ok, flat_ratio, white_ratio


# ---------------- Sidebar settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    min_conf = st.slider("Minimum confidence (%)", 30, 95, 60,
                         help="If the model is less sure than this, it will say 'not sure' instead of guessing.")
    use_check = st.checkbox("Reject screenshots / non-photos", value=True)
    st.markdown("---")
    st.caption("This model only knows the foods it was trained on. "
               "Anything else will be forced into one of those foods.")

# ---------------- Main layout ----------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### 1. Upload image")
    uploaded = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    img_rgb = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read this image. Please try another file.")
        else:
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)

    st.markdown("#### 2. Portion size (optional)")
    know_portion = st.checkbox("I know the portion size")
    grams = None
    if know_portion:
        grams = st.number_input("Portion size in grams", min_value=1, max_value=2000,
                                value=100, step=10)
    predict_clicked = st.button("🔍 Predict Food", type="primary",
                                use_container_width=True, disabled=img_rgb is None)

with right:
    st.markdown("#### 3. Result")

    if img_rgb is None:
        st.info("Upload a food image on the left to get started.")

    elif predict_clicked:
        # Step A: reject non-photos
        is_photo, flat_ratio, white_ratio = photo_check(img_rgb)
        if use_check and not is_photo:
            st.error("❌ This does not look like a food photo (it looks like a screenshot "
                     "or document). Please upload a real food picture.")
            st.caption(f"Check values - flat area: {flat_ratio:.0%}, white area: {white_ratio:.0%}")
            st.stop()

        # Step B: predict
        food, confidence, probs, class_names = predict_food(img_rgb)

        # Step C: low confidence -> do not guess
        if confidence < min_conf:
            st.warning(f"🤔 Not sure what this is (best guess: {pretty_name(food)}, "
                       f"only {confidence:.1f}% confident). Try a clearer photo of one food item.")
            st.stop()

        # Step D: show result
        result = estimate_calories(food, grams)

        st.markdown(f"""
        <div class="card">
          <div class="label">Detected food</div>
          <div class="food-name">{pretty_name(food)}</div>
          <div class="label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(confidence), 100), text=f"{confidence:.1f}%")

        if result:
            kcal, used = result
            note = (f"for your portion of {used} g" if grams
                    else f"for a typical serving of {used} g")
            st.markdown(f"""
            <div class="card">
              <div class="label">Estimated calories</div>
              <div class="kcal">{kcal} <small>kcal</small></div>
              <div class="label" style="margin-top:6px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No calorie data for this food. Add it to calories.py")

        # Top-3 predictions
        top3 = np.argsort(probs)[::-1][:3]
        chips = "".join(f'<span class="chip">{pretty_name(class_names[i])} {probs[i] * 100:.1f}%</span>'
                        for i in top3)
        st.markdown(f'<div class="label">Top 3 predictions</div>{chips}', unsafe_allow_html=True)

        with st.expander("Show all probabilities"):
            chart = pd.DataFrame({"Probability (%)": probs * 100},
                                 index=[pretty_name(c) for c in class_names])
            st.bar_chart(chart)

        st.caption(f"ℹ️ {DISCLAIMER}")
