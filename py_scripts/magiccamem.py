import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load your dataset
df = pd.read_csv("orimossimmos.csv")

# Load CamemBERT
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # inference mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get sentence embedding (CLS token from CamemBERT)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().cpu().numpy()

# Extract embeddings for all sentence pairs
embedding_diffs = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CamemBERT embeddings"):
    emb_ori = get_embedding(row["original_sentence"])
    emb_sim = get_embedding(row["simplified_sentence"])
    diff = emb_sim - emb_ori
    embedding_diffs.append(diff)

# Convert to DataFrame
embedding_dim = embedding_diffs[0].shape[0]
col_names = [f"embed_diff_{i}" for i in range(embedding_dim)]
df_embed = pd.DataFrame(embedding_diffs, columns=col_names)

# Save
df_embed.to_csv("embedding_diff.csv", index=False)
print("✅ Embedding differences saved to embedding_diff.csv")
