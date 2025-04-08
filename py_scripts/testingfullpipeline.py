import torch
from transformers import CamembertTokenizer, CamembertModel
import joblib
import numpy as np
import pandas as pd
import pyphen
import spacy
import sys
from pathlib import Path

# === Load assets ===
model = joblib.load("model_mlp_01.pkl")
scaler = joblib.load("scaler_norma_mlp_01.pkl")  # your fitted StandardScaler
nlp = spacy.load("fr_core_news_sm")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")

# === Import readability ===
sys.path.insert(0, r'C:\Users\marva\Desktop\Root\memoire\readability')
import readability


# === CamemBERT embedding extraction ===
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = camembert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # (768,)


# === Readability feature extraction ===
def get_readability_features(text, lang='fr'):
    doc = nlp(text)
    tokenized = '\n\n'.join(' '.join(token.text for token in sent) for sent in doc.sents)
    results = readability.getmeasures(tokenized, lang=lang, merge=True)
    for key in ['Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase',
                'GunningFogIndex', 'SMOGIndex', 'DaleChallIndex',
                'paragraphs', 'complex_words_dc']:
        results.pop(key, None)
    return results


# === Predict function ===
def predict_gain(original, simplified):
    # Embedding difference
    emb_ori = get_cls_embedding(original)
    emb_sim = get_cls_embedding(simplified)
    emb_diff = emb_sim - emb_ori  # (768,)

    # Readability feature difference
    feat_ori = get_readability_features(original)
    feat_sim = get_readability_features(simplified)

    feat_ori_df = pd.DataFrame([feat_ori]).add_prefix("ori_")
    feat_sim_df = pd.DataFrame([feat_sim]).add_prefix("sim_")
    feat_diff_array = feat_sim_df.values - feat_ori_df.values  # (1, N)

    # Only scale the handcrafted features
    feat_diff_scaled = scaler.transform([feat_diff_array.flatten()])  # shape (1, N)

    from sklearn.decomposition import PCA

    # # # Apply PCA to reduce embedding dimensions
    # pca = PCA(n_components=250)  # You can adjust this number (e.g., 20–100)
    # X_embed_pca = pca.fit_transform(emb_diff)
    #
    # #
    # # # Convert to DataFrame with proper column names
    # X_embed_pca_df = pd.DataFrame(X_embed_pca, columns=[f"pca_{i + 1}" for i in range(X_embed_pca.shape[1])])

    from sklearn.decomposition import PCA

    # ❌ Remove this (no more fitting at prediction time):
    # pca = PCA(n_components=250)
    # X_embed_pca = pca.fit_transform(emb_diff.reshape(1, -1))

    # ✅ Replace with this:
    pca = joblib.load("pca_model.pkl")
    X_embed_pca = pca.transform(emb_diff.reshape(1, -1))

    # Then combine scaled handcrafted + raw emb_diff
    all_features = np.concatenate([X_embed_pca.flatten(), feat_diff_scaled.flatten()])

    # Predict
    pred = model.predict([all_features])[0]
    return pred


# === Example usage ===
original = "Ceci est une phrase assez compliquée à comprendre pour certains lecteurs."
simplified = "Cette phrase est plus simple à lire."

gain = predict_gain(original, simplified)
print(f"📈 Readability gain (predicted): {gain:.3f}")
