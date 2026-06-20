import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Конфигурација странице ────────────────────────────────────────
st.set_page_config(
    page_title="Класификатор метал поджанрова",
   #page_icon="🤘",
    layout="centered"
)

# ─── Учитавање модела и помоћних објеката ──────────────────────────
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

#GENRE_EMOJI = {
  #  "black-metal": "🦇",
   # "death-metal": "💀",
   # "symphonic-metal": "🎻",
   # "thrash-metal": "⚡"
#}

# Тоналитети – Spotify key кодира тонове бројевима 0-11
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_OPTIONS = list(range(12))

# ─── Наслов ─────────────────────────────────────────────────────────
st.title("Класификатор Metal поджанрова")
st.markdown(
    "Унеси аудио карактеристике пјесме и модел ће предвидјети којем "
    "**метал поджанру** највјероватније припада: **black, death, symphonic** или **thrash metal**."
)
st.divider()

# ─── Улазни атрибути ────────────────────────────────────────────────
st.subheader("️Аудио карактеристике")

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
    valence = st.slider("Valence (веселост)", 0.0, 1.0, 0.25, 0.01)
    tempo = st.slider("Tempo (BPM)", 40, 220, 130)
    duration_s = st.slider("Трајање (секунде)", 60, 600, 240)
    key = st.selectbox("Key (тоналитет)", KEY_OPTIONS, index=0, format_func=lambda x: KEY_NAMES[x])
    mode = st.radio("Mode", options=[1, 0], format_func=lambda x: "Дур (Major)" if x == 1 else "Мол (Minor)")
    explicit = st.radio("Експлицитан садржај", options=[0, 1], format_func=lambda x: "Не" if x == 0 else "Да")
    time_signature = st.selectbox("Time signature", [3, 4, 5], index=1)

st.divider()

# ─── Предикција ─────────────────────────────────────────────────────
if st.button("Предвиди поджанр", use_container_width=True, type="primary"):

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

    # Скалирање истим scaler-ом као у тренингу
    input_scaled = scaler.transform(input_data)

    # Предикција
    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    pred_genre = label_encoder.inverse_transform([pred_encoded])[0]

    #moji = GENRE_EMOJI.get(pred_genre, "🎵")

    st.success(f"Предвиђени поджанр: **{pred_genre.upper()}**")

    st.markdown("#### Вјероватноће по жанру")
    proba_df = pd.DataFrame({
        "Жанр": label_encoder.classes_,
        "Вјероватноћа (%)": (pred_proba * 100).round(2)
    }).sort_values("Вјероватноћа (%)", ascending=False)

    st.bar_chart(proba_df.set_index("Жанр"))
    st.dataframe(proba_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Модел: XGBoost (тjунован путем RandomizedSearchCV) · "
    "Accuracy: 83,2% · F1-score: 0,834 · "
    "Предмет: Софтверски алгоритми у системима аутоматског управљања"
)