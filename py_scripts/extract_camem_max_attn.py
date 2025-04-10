import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

df = pd.read_csv("sentence_pairs.csv")

model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pooling Methods

def get_attention_weighted_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        attention_mask = inputs['attention_mask']  # [1, seq_len]

        token_embeddings = last_hidden_state.squeeze(0)
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        token_norms = torch.norm(token_embeddings * mask_expanded.squeeze(0), dim=1, keepdim=True)
        attention_weights = token_norms / token_norms.sum().clamp(min=1e-9)

        weighted_embeddings = token_embeddings * attention_weights * mask_expanded.squeeze(0)
        pooled_embedding = weighted_embeddings.sum(dim=0)

    return pooled_embedding.cpu().numpy()

def get_max_pooled_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        attention_mask = inputs['attention_mask']  # [1, seq_len]

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        token_embeddings = last_hidden_state * mask_expanded

        token_embeddings[mask_expanded == 0] = -1e9

        max_pooled = torch.max(token_embeddings, dim=1)[0]

    return max_pooled.squeeze().cpu().numpy()

# Extract Embeddings

attn_embedding_diffs = []
max_embedding_diffs = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Pooling CamemBERT embeddings"):
    # Attention-weighted pooling
    emb_ori_attn = get_attention_weighted_embedding(row["original_sentence"])
    emb_sim_attn = get_attention_weighted_embedding(row["simplified_sentence"])
    diff_attn = emb_sim_attn - emb_ori_attn
    attn_embedding_diffs.append(diff_attn)

    # Max pooling
    emb_ori_max = get_max_pooled_embedding(row["original_sentence"])
    emb_sim_max = get_max_pooled_embedding(row["simplified_sentence"])
    diff_max = emb_sim_max - emb_ori_max
    max_embedding_diffs.append(diff_max)

# Saving
embedding_dim = attn_embedding_diffs[0].shape[0]
col_names = [f"embed_attn_diff_{i}" for i in range(embedding_dim)]
df_embed_attn = pd.DataFrame(attn_embedding_diffs, columns=col_names)

embedding_dim = max_embedding_diffs[0].shape[0]
col_names = [f"embed_max_diff_{i}" for i in range(embedding_dim)]
df_embed_max = pd.DataFrame(max_embedding_diffs, columns=col_names)

df_embed_attn.to_csv("embedding_diff_attn_pool.csv", index=False)
df_embed_max.to_csv("embedding_diff_max_pool.csv", index=False)