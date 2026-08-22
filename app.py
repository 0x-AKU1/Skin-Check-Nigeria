"""
SkinCheck Nigeria (Streamlit version)
--------------------------------------
Upload a photo of a skin concern -> get a preliminary AI screening
-> get referred to the nearest partner clinics in Nigeria.

This is a SCREENING AID, not a diagnosis. It is designed to lower the
barrier to "should I see someone about this?" and point people toward
real care, not to replace a dermatologist.

Model: Jayanth2002/dinov2-base-finetuned-SkinDisease (pretrained, not
fine-tuned further here). ~22-31 dermatological classes, ~96% reported
test accuracy on its own benchmark.

Run locally:      streamlit run app.py
Deploy free:       push to GitHub -> share.streamlit.io -> connect repo
"""

import math
import csv
import os
import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

st.set_page_config(page_title="SkinCheck Nigeria", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# 1. MODEL (cached so it only loads once, not on every interaction)
# ---------------------------------------------------------------------------

MODEL_ID = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

# Optional second model covering conditions the main model is missing
# (Vitiligo, Eczema, Contact Dermatitis, Scabies, Normal Skin). Leave this
# blank until you've trained one with train_specialist_colab.py and pushed
# it to your own Hugging Face repo — the app runs fine without it, it just
# won't catch these specific conditions until you add it.
SPECIALIST_MODEL_ID = "1nOnlyVic/skincheck-specialist"


@st.cache_resource(show_spinner="Loading model (first run only, ~350MB)...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    model.eval()
    torch.set_num_threads(1)
    return processor, model


@st.cache_resource(show_spinner="Loading specialist model...")
def load_specialist_model():
    if not SPECIALIST_MODEL_ID:
        return None, None
    processor = AutoImageProcessor.from_pretrained(SPECIALIST_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(SPECIALIST_MODEL_ID, low_cpu_mem_usage=True)
    model.eval()
    return processor, model


processor, model = load_model()
specialist_processor, specialist_model = load_specialist_model()
ID2LABEL = model.config.id2label

URGENT_KEYWORDS = [
    "melanoma", "carcinoma", "malignant", "leprosy", "lupus",
    "epidermolysis", "neurofibromatosis", "monkeypox",
]


def urgency_for(label: str) -> str:
    low = label.lower()
    if any(k in low for k in URGENT_KEYWORDS):
        return "high"
    return "routine"


def is_normal_skin(label: str) -> bool:
    return "normal" in label.lower() and "skin" in label.lower()


def predict_skin(image):
    img = image.convert("RGB")
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    # If a specialist model is configured, run it too and merge results into
    # one ranked list — so "Vitiligo" or "Eczema" can outrank a main-model
    # guess if the specialist model is more confident about it.
    combined = [(ID2LABEL[i], float(p)) for i, p in enumerate(probs)]

    if specialist_model is not None:
        sp_inputs = specialist_processor(img, return_tensors="pt")
        with torch.no_grad():
            sp_logits = specialist_model(**sp_inputs).logits
        sp_probs = torch.softmax(sp_logits, dim=-1)[0]
        sp_id2label = specialist_model.config.id2label
        # Halve specialist confidences slightly relative to the main model's
        # scale isn't necessary since both are independent softmaxes over
        # different class sets — we just merge and re-rank by raw confidence.
        combined += [(sp_id2label[i], float(p)) for i, p in enumerate(sp_probs)]

    combined.sort(key=lambda x: x[1], reverse=True)
    results = combined[:5]
    top_label = results[0][0]
    return results, urgency_for(top_label)


# ---------------------------------------------------------------------------
# 2. CLINICS
# ---------------------------------------------------------------------------

CLINICS_CSV = os.path.join(os.path.dirname(__file__), "clinics.csv")

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu",
    "FCT (Abuja)", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo",
    "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
]


@st.cache_data
def load_clinics():
    clinics = []
    if not os.path.exists(CLINICS_CSV):
        return clinics
    with open(CLINICS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clinics.append(row)
    return clinics


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_clinics(state: str, city: str, limit=5):
    clinics = load_clinics()
    if not clinics:
        return []
    same_state = [c for c in clinics if c.get("state", "").strip().lower() == (state or "").strip().lower()]
    pool = same_state if same_state else clinics
    if city:
        pool = sorted(pool, key=lambda c: 0 if c.get("city", "").strip().lower() == city.strip().lower() else 1)
    return pool[:limit]


# ---------------------------------------------------------------------------
# 3. UI
# ---------------------------------------------------------------------------

st.title("🩺 SkinCheck Nigeria")
st.warning(
    "**This is an AI screening aid, not a medical diagnosis.** It cannot examine "
    "your skin the way a clinician can, and it can be wrong. Use it to decide "
    "*whether to see someone* — not as a final answer. If you notice rapid "
    "change, bleeding, spreading, or severe pain, see a clinic now regardless "
    "of what this tool says."
)

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Photo of the affected skin area", type=["png", "jpg", "jpeg"])
    state = st.selectbox("Your state", NIGERIAN_STATES, index=NIGERIAN_STATES.index("Lagos"))
    city = st.text_input("Your city / area (optional)", placeholder="e.g. Ikeja")
    analyze = st.button("Analyze", type="primary")

with col2:
    if analyze and uploaded is not None:
        from PIL import Image
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded photo", width=250)

        with st.spinner("Analyzing..."):
            results, urgency = predict_skin(image)

        st.subheader("Top possibilities")
        for label, prob in results:
            st.write(f"**{label}** — {prob*100:.1f}%")
            st.progress(min(prob, 1.0))
        st.caption(
            "These are ranked guesses from an image classifier, not a diagnosis. "
            "Multiple skin conditions look alike in photos."
        )

        if is_normal_skin(results[0][0]) and results[0][1] > 0.5:
            st.success(
                "✅ This photo doesn't show a clear sign of any of the conditions "
                "this tool screens for. If something still feels off (itching, "
                "pain, changes over time), it's still worth a professional look — "
                "this isn't a clean bill of health, just a low-signal result."
            )
        elif urgency == "high":
            st.error(
                "🔴 One of the top matches is a condition that can be serious "
                "(e.g. malignant or systemic). Please prioritize seeing a "
                "clinician in person soon — don't wait this one out."
            )

        st.subheader(f"Suggested clinics near {state}")
        clinics = find_clinics(state, city)
        if not clinics:
            st.info(f"No clinics loaded yet for {state}.")
        else:
            for c in clinics:
                with st.container(border=True):
                    st.markdown(f"**{c.get('name','Unnamed clinic')}**")
                    st.write(f"{c.get('address','')}, {c.get('city','')}, {c.get('state','')}")
                    st.write(f"📞 {c.get('phone','N/A')}")
    elif analyze and uploaded is None:
        st.info("Upload a photo first.")
    else:
        st.info("Upload a clear, well-lit photo and click Analyze to get started.")

st.markdown("---")
st.caption(
    f"Model: [Jayanth2002/dinov2-base-finetuned-SkinDisease]"
    f"(https://huggingface.co/{MODEL_ID}). Clinic directory is community-maintained — see clinics.csv."
)
