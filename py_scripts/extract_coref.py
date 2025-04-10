import pandas as pd
import spacy
import coreferee

nlp = spacy.load('fr_core_news_lg')
nlp.add_pipe('coreferee')

def get_coref_features(text):
        doc = nlp(text)
        chains = doc._.coref_chains

        return {
                "has_coref_chain": int(len(chains) > 0),
                "n_pronouns": sum(1 for token in doc if token.pos_ == "PRON")
        }

df = pd.read_csv("sentence_pairs.csv")

# Extract coreference features for both original and simplified sentences
original_features = df["original_sentence"].apply(get_coref_features)
simplified_features = df["simplified_sentence"].apply(get_coref_features)

df_original = pd.DataFrame(original_features.tolist())
df_simplified = pd.DataFrame(simplified_features.tolist())

df_original = df_original.add_prefix("orig_")
df_simplified = df_simplified.add_prefix("simp_")

df_coref = pd.concat([df, df_original, df_simplified], axis=1)
df_coref.to_csv("coref_features.csv", index=False)

# Compute differences between original and simplified
df_coref["diff_n_pronouns"] = df_coref["simp_n_pronouns"] - df_coref["orig_n_pronouns"]
df_coref["diff_has_coref_chain"] = df_coref["simp_has_coref_chain"] - df_coref["orig_has_coref_chain"]

# Binary flag: original had a chain but it disappeared in simplification
df_coref["coref_chain_removed"] = (
    (df_coref["orig_has_coref_chain"] == 1) &
    (df_coref["simp_has_coref_chain"] == 0)
).astype(int)

df_coref.to_csv("coref_features_with_diff.csv", index=False)

final_columns = ["diff_n_pronouns", "diff_has_coref_chain", "coref_chain_removed"]
df_coref_clean = df_coref[final_columns]
df_coref_clean.to_csv("X_coref.csv", index=False)

