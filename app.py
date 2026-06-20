import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Konfiguracija stranice ────────────────────────────────────────
st.set_page_config(
    page_title="Metal Genre Classifier",
    page_icon="🤘",
    layout="centered"
)

# ─── Učitavanje modela i pomoćnih objekata ─────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_tuned.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

FEATURE_COLS = [
    "popularity", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "time_signature", "duration_s"
]

GENRE_EMOJI = {
    "black-metal": "🦇",
    "death-metal": "💀",
    "symphonic-metal": "🎻",
    "thrash-metal": "⚡"
}

# ─── Naslov ─────────────────────────────────────────────────────────
st.title("🤘 Metal Genre Classifier")
st.markdown(
    "Unesi audio karakteristike pjesme i model će predvidjeti kojem "
    "**metal podžanru** najvjerovatnije pripada: **black, death, symphonic** ili **thrash metal**."
)
st.divider()

# ─── Ulazni atributi ────────────────────────────────────────────────
st.subheader("🎚️ Audio karakteristike")

col1, col2 = st.columns(2)

with col1:
    popularity = st.slider("Popularity", 0, 100, 50)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.4, 0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.85, 0.01)
    loudness = st.slider("Loudness (dB)", -30.0, 0.0, -6.0, 0.5)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.08, 0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.02, 0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.01)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.01)

with col2:
    valence = st.slider("Valence (veselost)", 0.0, 1.0, 0.25, 0.01)
    tempo = st.slider("Tempo (BPM)", 40, 220, 130)
    duration_s = st.slider("Trajanje (sekunde)", 60, 600, 240)
    key = st.selectbox("Key (tonalitet)", list(range(0, 12)), index=0)
    mode = st.radio("Mode", options=[1, 0], format_func=lambda x: "Dur (Major)" if x == 1 else "Mol (Minor)")
    explicit = st.radio("Explicit sadržaj", options=[0, 1], format_func=lambda x: "Ne" if x == 0 else "Da")
    time_signature = st.selectbox("Time signature", [3, 4, 5], index=1)

st.divider()

# ─── Predikcija ─────────────────────────────────────────────────────
if st.button("🔮 Predvidi podžanr", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "popularity": popularity,
        "explicit": explicit,
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "time_signature": time_signature,
        "duration_s": duration_s
    }])[FEATURE_COLS]

    # Skaliranje istim scalerom kao u treningu
    input_scaled = scaler.transform(input_data)

    # Predikcija
    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    pred_genre = label_encoder.inverse_transform([pred_encoded])[0]

    emoji = GENRE_EMOJI.get(pred_genre, "🎵")

    st.success(f"### {emoji} Predviđeni podžanr: **{pred_genre.upper()}**")

    st.markdown("#### Vjerovatnoće po žanru")
    proba_df = pd.DataFrame({
        "Žanr": label_encoder.classes_,
        "Vjerovatnoća (%)": (pred_proba * 100).round(2)
    }).sort_values("Vjerovatnoća (%)", ascending=False)

    st.bar_chart(proba_df.set_index("Žanr"))
    st.dataframe(proba_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Model: XGBoost (tunovan putem RandomizedSearchCV) · "
    "Accuracy: 83.2% · F1-score: 0.834 · "
    "Predmet: Softverski algoritmi u sistemima automatskog upravljanja"
)
