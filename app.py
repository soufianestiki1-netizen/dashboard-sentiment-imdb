"""
Dashboard de Détection de Sentiment — Avis de Films (IMDB)
Module BDA25706 — AI & Machine Learning | Ravensbourne University London

Version simplifiée pour déploiement sur Streamlit Community Cloud.
Le modèle LSTM tourne en direct ; DistilBERT est présenté via ses résultats déjà mesurés
(le modèle complet est trop volumineux pour ce déploiement léger).
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(
    page_title="Détection de Sentiment - Avis de Films",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Chargement du modèle LSTM (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_lstm_model():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.datasets import imdb
    model = load_model('baseline_lstm_imdb.h5')
    word_index = imdb.get_word_index()
    return model, word_index

VOCAB_SIZE = 10000
MAX_LEN = 300

def encode_for_lstm(text, word_index, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    words = text.split()
    seq = []
    for w in words:
        idx = word_index.get(w)
        if idx is not None and idx + 3 < vocab_size:
            seq.append(idx + 3)
        else:
            seq.append(2)
    padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')
    return padded

def predict_lstm(text, model, word_index):
    padded = encode_for_lstm(text, word_index)
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Positif" if prob > 0.5 else "Négatif"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("🎬 Détection de Sentiment sur Avis de Films")
st.markdown("""
**Problème business** : les plateformes de streaming et e-commerce reçoivent des milliers d'avis clients
chaque jour. Ce système classe automatiquement chaque avis comme **positif** ou **négatif**, permettant
une analyse de satisfaction à grande échelle, sans lecture manuelle.
""")

tab1, tab2, tab3 = st.tabs(["🔮 Prédiction en direct", "📊 Comparaison des modèles", "ℹ️ À propos du projet"])

# --- Onglet 1 : Prédiction en direct (LSTM) ---
with tab1:
    st.subheader("Teste le modèle LSTM sur un nouvel avis")
    st.caption("Modèle baseline en direct — le modèle DistilBERT (technique avancée) est présenté avec ses résultats mesurés dans l'onglet 'Comparaison des modèles'.")

    sample_reviews = {
        "-- Choisir un exemple ou écrire le tien --": "",
        "Exemple positif": "This movie completely blew me away. The acting was superb, the plot kept me engaged from start to finish, and the cinematography was stunning.",
        "Exemple négatif": "What a waste of time. The plot made no sense, the acting felt wooden, and I found myself checking my phone every ten minutes out of boredom.",
        "Exemple positif subtil": "Not perfect, but the performances carried the film through its slower moments. By the end I was genuinely moved.",
        "Exemple négatif subtil": "I wanted to like this more than I did. Despite a promising premise, the pacing dragged and the ending felt rushed.",
    }

    choice = st.selectbox("Charger un exemple (optionnel) :", list(sample_reviews.keys()))
    default_text = sample_reviews[choice]

    user_text = st.text_area(
        "Colle ou écris un avis de film (en anglais) :",
        value=default_text,
        height=150,
        placeholder="Ex: This film was absolutely wonderful, the story..."
    )

    if st.button("🔍 Analyser le sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Merci d'écrire ou de choisir un avis avant de lancer l'analyse.")
        elif not os.path.exists('baseline_lstm_imdb.h5'):
            st.error("Fichier 'baseline_lstm_imdb.h5' introuvable dans ce dépôt.")
        else:
            with st.spinner("Analyse en cours..."):
                lstm_model, word_index = load_lstm_model()
                label_l, conf_l, prob_l = predict_lstm(user_text, lstm_model, word_index)

            color = "green" if label_l == "Positif" else "red"
            st.markdown(f"### Prédiction : <span style='color:{color}'>{label_l}</span>", unsafe_allow_html=True)
            st.progress(conf_l)
            st.caption(f"Confiance : {conf_l:.1%}")

# --- Onglet 2 : Comparaison des modèles ---
with tab2:
    st.subheader("Performance comparée : Baseline vs Technique Avancée")

    if os.path.exists('model_comparison.csv'):
        comp_df = pd.read_csv('model_comparison.csv')
        st.dataframe(comp_df, use_container_width=True)
        st.bar_chart(comp_df.set_index('Modèle')['Test Accuracy'])
    else:
        default_comp = pd.DataFrame({
            'Modèle': ['LSTM Baseline', 'DistilBERT Fine-tuné'],
            'Test Accuracy': [0.8465, 0.8972]
        })
        st.dataframe(default_comp, use_container_width=True)
        st.bar_chart(default_comp.set_index('Modèle')['Test Accuracy'])

    st.markdown("""
    **Interprétation** : le passage d'un LSTM entraîné from scratch à un Transformer pré-entraîné
    (DistilBERT) apporte un gain de **+5 points d'accuracy**, illustrant l'avantage du transfer learning
    en NLP — le modèle bénéficie d'une compréhension du langage déjà acquise sur un immense corpus de texte.

    *Le modèle DistilBERT complet (~250 Mo) n'est pas chargé dans cette version déployée du dashboard
    pour des raisons de taille, mais son code d'entraînement et d'évaluation complet est disponible
    dans le notebook du projet.*
    """)

# --- Onglet 3 : À propos ---
with tab3:
    st.subheader("À propos de ce projet")
    st.markdown("""
    **Module** : BDA25706 — Artificial Intelligence and Machine Learning
    **Établissement** : Ravensbourne University London

    **Dataset** : IMDB Movie Reviews (Maas et al., 2011) — 50 000 avis, référence académique en NLP.

    **Pipeline** :
    1. Modèle baseline : LSTM bidirectionnel entraîné from scratch (déployé en direct ici)
    2. Modèle avancé : DistilBERT fine-tuné (Transformer pré-entraîné) — voir notebook complet
    3. Évaluation complète : accuracy, precision/recall/F1, ROC-AUC, matrice de confusion
    4. Validation externe sur des avis rédigés spécifiquement pour ce projet

    **Limites connues** :
    - Le modèle est entraîné sur des avis en anglais uniquement
    - Les phrases très ironiques ou au second degré peuvent être mal interprétées
    - Le dataset d'entraînement (IMDB) exclut volontairement les avis neutres/mitigés,
      donc le modèle n'a jamais appris à détecter un sentiment ambivalent
    """)

st.markdown("---")
st.caption("Projet académique — BDA25706 AI & Machine Learning — Ravensbourne University London")
