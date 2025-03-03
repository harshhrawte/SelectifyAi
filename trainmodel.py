import pandas as pd
import spacy
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import RobertaTokenizer, RobertaModel  # Use RoBERTa for better accuracy
import gc

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset in chunks to save memory
dataset_path = "static/UpdatedResumeDataSet.csv"
chunksize = 500  # Smaller chunks to reduce RAM usage
df = pd.concat([chunk for chunk in pd.read_csv(dataset_path, chunksize=chunksize)])

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unnecessary components for faster processing

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # Use RoBERTa
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):  # Handle NaN or non-string values
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Apply preprocessing in smaller batches
batch_size = 500  # Smaller batch size to reduce RAM usage
processed_resumes = []
for i in range(0, len(df), batch_size):
    batch = df["Resume"][i:i + batch_size].tolist()
    processed_resumes.extend([preprocess_text(text) for text in batch])
    del batch
    gc.collect()

# Assign processed resumes to the DataFrame
df["Processed_Resume"] = processed_resumes

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)  # Reduce max_features to save memory
X_tfidf = vectorizer.fit_transform(df["Processed_Resume"])
y = df["Category"]

# Train Logistic Regression Model
model = LogisticRegression(n_jobs=-1)  # Utilize all CPU cores
model.fit(X_tfidf, y)

# Function to get RoBERTa embeddings
def get_roberta_embedding(text):
    if not text.strip():  # Handle empty texts
        return torch.zeros(768).numpy()  # Default zero vector
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)  # Reduce max_length to 256
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Compute RoBERTa embeddings in smaller batches
def compute_roberta_embeddings(texts, batch_size=8):  # Smaller batch size for GPU
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = [get_roberta_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
        del batch, batch_embeddings
        gc.collect()
    return embeddings

df["BERT_Embedding"] = compute_roberta_embeddings(df["Processed_Resume"])

# Save models
with open("static/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("static/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

df.to_pickle("static/bert_embeddings.pkl")  # Save RoBERTa embeddings separately

print("Model, vectorizer, and RoBERTa embeddings saved successfully.")