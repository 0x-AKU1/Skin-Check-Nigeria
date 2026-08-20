"""
SkinCheck Nigeria
------------------
Upload a photo of a skin concern -> get a preliminary AI screening
-> get referred to the nearest partner clinics in Nigeria.

This is a SCREENING AID, not a diagnosis. It is designed to lower the
barrier to "should I see someone about this?" and point people toward
real care, not to replace a dermatologist.

Model: Jayanth2002/dinov2-base-finetuned-SkinDisease (pretrained, not
fine-tuned further here). ~22-31 dermatological classes, ~96% reported
test accuracy on its own benchmark. See MODEL_CARD.md for details.

Run locally:      python app.py
Deploy on HF:      push this repo to a Hugging Face Space (see README.md)
"""

import math
import csv
import os
import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

# ---------------------------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------------------------

MODEL_ID = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

print("Loading model... (first run downloads weights, ~350MB)")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

ID2LABEL = model.config.id2label

# Very rough "what kind of condition is this" bucket, used only to decide
# how urgent the referral message sounds. Keep this conservative: anything
# not obviously benign defaults to "see a professional soon."
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
    if image is None:
        return [], "routine"
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
# clinics.csv columns: name,state,city,address,phone,lat,lon
# lat/lon are OPTIONAL. If present for both the clinic and the user's
# chosen location, we sort by real distance (haversine). If not, we just
# group by state/city.
#
# Vic: drop your real clinic list into clinics.csv using this header and
# everything below just works. A tiny starter file ships here so the app
# runs end-to-end before you send the real data.

CLINICS_CSV = os.path.join(os.path.dirname(__file__), "clinics.csv")

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu",
    "FCT (Abuja)", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo",
    "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
]


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


def find_clinics(state: str, city: str, user_lat=None, user_lon=None, limit=5):
    clinics = load_clinics()
    if not clinics:
        return []

    same_state = [c for c in clinics if c.get("state", "").strip().lower() == (state or "").strip().lower()]
    pool = same_state if same_state else clinics  # fall back to nationwide list if nothing in-state

    if user_lat is not None and user_lon is not None:
        def dist(c):
            try:
                return haversine_km(user_lat, user_lon, float(c["lat"]), float(c["lon"]))
            except (KeyError, ValueError, TypeError):
                return float("inf")
        pool = sorted(pool, key=dist)
    else:
        # no coordinates: prioritize same city, then just keep CSV order
        if city:
            pool = sorted(pool, key=lambda c: 0 if c.get("city", "").strip().lower() == city.strip().lower() else 1)

    return pool[:limit]


# ---------------------------------------------------------------------------
# 3. APP LOGIC
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "⚠️ **This is an AI screening aid, not a medical diagnosis.** "
    "It cannot examine your skin the way a clinician can, and it can be wrong. "
    "Use it to decide *whether to see someone* — not as a final answer. "
    "If you notice rapid change, bleeding, spreading, or severe pain, see a "
    "clinic now regardless of what this tool says."
)


def format_predictions(results):
    if not results:
        return "Upload a clear, well-lit photo of the affected area to get started."
    lines = ["### Top possibilities\n"]
    for label, prob in results:
        bar = "█" * max(1, int(prob * 20))
        lines.append(f"**{label}** — {prob*100:.1f}%\n`{bar}`\n")
    lines.append(
        "\n*These are ranked guesses from an image classifier, not a diagnosis. "
        "Multiple skin conditions look alike in photos.*"
    )
    return "\n".join(lines)


def format_clinics(clinics, state):
    if not clinics:
        return (
            f"No clinics loaded yet for **{state or 'your area'}**. "
            "Once the clinic directory is added to `clinics.csv`, matches will appear here."
        )
    lines = [f"### Suggested clinics near **{state}**\n"]
    for c in clinics:
        lines.append(
            f"**{c.get('name','Unnamed clinic')}**  \n"
            f"{c.get('address','')}, {c.get('city','')}, {c.get('state','')}  \n"
            f"📞 {c.get('phone','N/A')}\n"
        )
    return "\n".join(lines)


def run(image, state, city):
    results, urgency = predict_skin(image)
    pred_md = format_predictions(results)

    urgency_note = ""
    if urgency == "high":
        urgency_note = (
            "\n\n🔴 **One of the top matches is a condition that can be serious "
            "(e.g. malignant or systemic). Please prioritize seeing a clinician "
            "in person soon — don't wait this one out.**"
        )

    clinics = find_clinics(state, city)
    clinic_md = format_clinics(clinics, state)

    return pred_md + urgency_note, clinic_md


# ---------------------------------------------------------------------------
# 4. UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="SkinCheck Nigeria", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# 🩺 SkinCheck Nigeria")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Photo of the affected skin area")
            state_in = gr.Dropdown(choices=NIGERIAN_STATES, label="Your state", value="Lagos")
            city_in = gr.Textbox(label="Your city / area (optional)", placeholder="e.g. Ikeja")
            submit = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            pred_out = gr.Markdown()
            clinic_out = gr.Markdown()

    submit.click(run, inputs=[image_in, state_in, city_in], outputs=[pred_out, clinic_out])

    gr.Markdown(
        "---\n*Model: [Jayanth2002/dinov2-base-finetuned-SkinDisease]"
        "(https://huggingface.co/Jayanth2002/dinov2-base-finetuned-SkinDisease). "
        "Clinic directory is community-maintained — see `clinics.csv`.*"
    )

if __name__ == "__main__":
    # Render (and most non-HF hosts) assign a port via the PORT env var and
    # expect the app to bind to 0.0.0.0. Falls back to Gradio's default
    # (7860) for local runs, so nothing changes when testing on your machine.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
