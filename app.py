from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
import pickle
import PyPDF2
import docx
import spacy
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel  # Use RoBERTa
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
from waitress import serve
from multiprocessing import cpu_count
import gc

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
with open("static/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("static/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
bert_embeddings = pd.read_pickle("static/bert_embeddings.pkl")

# Load NLP tools
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unnecessary components for faster processing
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # Use RoBERTa
roberta_model = RobertaModel.from_pretrained("roberta-base").to("cuda" if torch.cuda.is_available() else "cpu")

uploaded_resumes = []

def preprocess_text(text):
    """Cleans and preprocesses text."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def extract_text(file_path, file_type):
    """Extracts text from PDFs and DOCX files."""
    try:
        if file_type == "pdf":
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                return preprocess_text(" ".join([page.extract_text() or "" for page in reader.pages]).strip())
        elif file_type == "docx":
            doc = docx.Document(file_path)
            return preprocess_text(" ".join([para.text for para in doc.paragraphs]).strip())
    except Exception as e:
        return f"Error reading {file_type.upper()}: {str(e)}"

def calculate_roberta_similarity(text1, text2):
    """Calculates RoBERTa-based cosine similarity between job description and resume."""
    try:
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")

        with torch.no_grad():
            embeddings1 = roberta_model(**inputs1).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings2 = roberta_model(**inputs2).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return cosine_similarity([embeddings1], [embeddings2])[0][0]
    except Exception as e:
        print("RoBERTa Similarity Error:", str(e))
        return 0.0

def match_skills(required_skills, resume_text):
    """Finds matching skills from resume."""
    resume_words = set(resume_text.lower().split())
    return [skill for skill in required_skills if any(fuzz.partial_ratio(skill.lower(), word) > 80 for word in resume_words)]

def find_best_match(resume_text):
    """Finds the best matching category using TF-IDF and RoBERTa embeddings."""
    tfidf_resume = vectorizer.transform([resume_text])
    category_pred = model.predict(tfidf_resume)[0]

    similar_resumes = bert_embeddings[bert_embeddings["Category"] == category_pred]["Processed_Resume"]
    
    if similar_resumes.empty:
        return category_pred, 0.0

    tfidf_similar_resumes = vectorizer.transform(similar_resumes)
    similarity_scores = cosine_similarity(tfidf_resume, tfidf_similar_resumes).flatten()

    return category_pred, max(similarity_scores) if len(similarity_scores) > 0 else 0.0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles resume upload and ranking."""
    global uploaded_resumes
    temp_uploaded_resumes = []

    if "resumes" not in request.files or "job_description" not in request.form or "skills" not in request.form:
        return jsonify({"error": "Provide resumes, job description, and skills."}), 400

    job_desc = preprocess_text(request.form["job_description"])
    required_skills = request.form["skills"].split(",")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Limit the number of workers to 6 (adjust as needed)
    max_workers = min(6, cpu_count())  # Use 6 cores to balance performance and heat
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in request.files.getlist("resumes"):
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            file_type = "pdf" if file.filename.endswith(".pdf") else "docx" if file.filename.endswith(".docx") else None

            if not file_type:
                return jsonify({"error": "Unsupported file format."}), 400

            futures.append(executor.submit(process_resume, file_path, file_type, job_desc, required_skills))

        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                return jsonify(result), 400
            temp_uploaded_resumes.append(result)

    temp_uploaded_resumes.sort(key=lambda x: x["final_score"], reverse=True)
    uploaded_resumes = temp_uploaded_resumes

    return redirect("/ranked-resume")

def process_resume(file_path, file_type, job_desc, required_skills):
    """Processes a single resume and computes scores."""
    resume_text = extract_text(file_path, file_type)
    if "Error" in resume_text:
        return {"error": resume_text}

    # Compute scores
    roberta_similarity = calculate_roberta_similarity(job_desc, resume_text)
    matched_skills = match_skills(required_skills, resume_text)
    skill_score = len(matched_skills) / len(required_skills) if required_skills else 0
    best_category, dataset_similarity = find_best_match(resume_text)
    final_score = (0.5 * roberta_similarity) + (0.4 * skill_score) + (0.1 * dataset_similarity)

    # Free up memory
    del resume_text, roberta_similarity, matched_skills, skill_score, best_category, dataset_similarity
    gc.collect()

    return {"name": os.path.basename(file_path), "final_score": final_score}

@app.route("/ranked-resume")
def ranked_resume():
    """Displays ranked resumes."""
    return render_template("ranked_resume.html", ranked_resumes=uploaded_resumes)

@app.route("/get-ranked-resumes", methods=["GET"])
def get_ranked_resumes():
    """Returns ranked resumes in JSON format."""
    return jsonify(uploaded_resumes)

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)