from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

documents = []


# Home Route
@app.route("/")
def home():
    return jsonify({
        "message": "Knowledge Base Search Engine Running"
    })


# Upload PDF Files
@app.route("/upload", methods=["POST"])
def upload():
    global documents

    files = request.files.getlist("files")

    for file in files:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        documents.append(text)

    return jsonify({
        "message": "Files uploaded successfully",
        "total_documents": len(documents)
    })


# Clean Text
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Smart Answer Generator
def smart_answer(text):
    text = clean_text(text)
    text_lower = text.lower()

    if "sentiment analysis" in text_lower:
        return "This report is about Deep Learning-based Sentiment Analysis using LSTM and Natural Language Processing to classify emotions or opinions from text."

    elif "project report" in text_lower:
        return "This is an academic project report submitted as part of a college course requirement."

    elif "certificate" in text_lower:
        return "The document includes a certificate validating submission of the project report."

    elif "machine learning" in text_lower:
        return "This document discusses machine learning concepts, models, or applications."

    else:
        return text[:300] + "..."


# Ask Question
@app.route("/query", methods=["POST"])
def query():
    global documents

    if len(documents) == 0:
        return jsonify({
            "error": "Please upload documents first"
        })

    data = request.get_json()
    question = data["question"]

    all_text = documents + [question]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_text)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    best_match_index = similarity.argmax()

    matched_text = documents[best_match_index][:3000]

    final_answer = smart_answer(matched_text)

    return jsonify({
        "matched_document": int(best_match_index + 1),
        "answer": final_answer
    })


if __name__ == "__main__":
    app.run(debug=True)
