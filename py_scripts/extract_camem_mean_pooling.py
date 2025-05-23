import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load your dataset
df = pd.read_csv("sentence_pairs.csv")

model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# mean pooling of token embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        mean_pooled = sum_embeddings / sum_mask.clamp(min=1e-9)

    return mean_pooled.squeeze().cpu().numpy()

# Computing embedding differences
embedding_diffs = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CamemBERT embeddings (mean pooling)"):
    emb_ori = get_embedding(row["original_sentence"])
    emb_sim = get_embedding(row["simplified_sentence"])
    diff = emb_sim - emb_ori
    embedding_diffs.append(diff)


embedding_dim = embedding_diffs[0].shape[0]
col_names = [f"embed_diff_{i}" for i in range(embedding_dim)]
df_embed = pd.DataFrame(embedding_diffs, columns=col_names)
df_embed.to_csv("embedding_diff_mean_pool.csv", index=False)