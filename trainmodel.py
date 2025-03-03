import pandas as pd
import spacy
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import RobertaTokenizer, RobertaModel  
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = "static/UpdatedResumeDataSet.csv"
chunksize = 500  
df = pd.concat([chunk for chunk in pd.read_csv(dataset_path, chunksize=chunksize)])
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)


def preprocess_text(text):
    if not isinstance(text, str):  
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

batch_size = 500  
processed_resumes = []
for i in range(0, len(df), batch_size):
    batch = df["Resume"][i:i + batch_size].tolist()
    processed_resumes.extend([preprocess_text(text) for text in batch])
    del batch
    gc.collect()

df["Processed_Resume"] = processed_resumes

vectorizer = TfidfVectorizer(max_features=3000)  
X_tfidf = vectorizer.fit_transform(df["Processed_Resume"])
y = df["Category"]


model = LogisticRegression(n_jobs=-1)  
model.fit(X_tfidf, y)


def get_roberta_embedding(text):
    if not text.strip():  
        return torch.zeros(768).numpy()  
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)  
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def compute_roberta_embeddings(texts, batch_size=8):  
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = [get_roberta_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
        del batch, batch_embeddings
        gc.collect()
    return embeddings

df["BERT_Embedding"] = compute_roberta_embeddings(df["Processed_Resume"])

with open("static/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("static/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

df.to_pickle("static/bert_embeddings.pkl")  

print("Model, vectorizer, and RoBERTa embeddings saved successfully.")
