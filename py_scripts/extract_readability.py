import pandas as pd
import numpy as np
import spacy
from pathlib import Path

# Load French model
nlp = spacy.load("fr_core_news_sm")

# !!! Load the *adapted readability package for French !!!
# Source: https://github.com/mervealgan/readability
# Make sure to adjust the path below for the 'readability' folder
import sys
sys.path.insert(0, r'C:\Users\marva\Desktop\Root\memoire\readability')
import readability

# Load the dataset
df = pd.read_csv("sentence_pairs.csv")

# Define the feature extraction function
def get_features(text, lang='fr'):
    doc = nlp(text)
    tokenized = '\n\n'.join(' '.join(token.text for token in sent) for sent in doc.sents)
    results = readability.getmeasures(tokenized, lang=lang, merge=True)

    for key in [
        'Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase',
        'GunningFogIndex', 'SMOGIndex', 'DaleChallIndex',
        'paragraphs', 'complex_words_dc'
    ]:
        results.pop(key, None)

    return results

# Applying to original and simplified
features_original = df['original_sentence'].apply(lambda x: get_features(x, 'fr'))
features_simplified = df['simplified_sentence'].apply(lambda x: get_features(x, 'fr'))

df_feat_ori = pd.DataFrame(features_original.tolist()).add_prefix("ori_")
df_feat_sim = pd.DataFrame(features_simplified.tolist()).add_prefix("sim_")

# Computing differences
df_feat_diff = df_feat_sim.values - df_feat_ori.values
df_feat_diff = pd.DataFrame(df_feat_diff, columns=[col.replace("sim_", "diff_") for col in df_feat_sim.columns])

# Saving
df_final = pd.concat([df, df_feat_ori, df_feat_sim, df_feat_diff], axis=1)

output_path = Path("X_read_with_sentences.csv")
df_final.to_csv(output_path, index=False)
