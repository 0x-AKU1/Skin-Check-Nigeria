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


@st.cache_resource(show_spinner="Loading model (first run only, ~350MB)...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    model.eval()
    torch.set_num_threads(1)
    return processor, model


processor, model = load_model()
ID2LABEL = model.config.id2label

URGENT_KEYWORDS = [
    "melanoma", "carcinoma", "malignant", "leprosy", "lupus",
    "epidermolysis", "neurofibromatosis",
]


def urgency_for(label: str) -> str:
    low = label.lower()
    if any(k in low for k in URGENT_KEYWORDS):
        return "high"
    return "routine"


def predict_skin(image):
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probs, k=min(5, probs.shape[-1]))
    results = [(ID2LABEL[i.item()], float(p)) for p, i in zip(top5.values, top5.indices)]
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

        if urgency == "high":
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
