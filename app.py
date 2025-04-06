from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import config
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
app.secret_key = 'supersecretkey'

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import joblib
tag_model = joblib.load("../models/tag_model.pkl")
tfidf_vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
mlb = joblib.load("../models/multilabel_binarizer.pkl")
engagement_model = joblib.load('../models/engagement.pkl')
keyword_vectorizer = joblib.load('../models/Keyword_Finder_tfidf_vectorizer.pkl')

df = pd.read_csv('../reduced_dataset.csv')
# MongoDB Setup
client = MongoClient(config.MONGO_URI)
db = client[config.DATABASE_NAME]
users = db[config.USER_COLLECTION]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if users.find_one({'email': email}):
            return "Email already exists!"

        users.insert_one({'name': name, 'email': email, 'password': password})
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            return redirect(url_for('main'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/main')
def main():
    if 'user' not in session:
        return redirect(url_for('login'))

    df = pd.read_csv('../reduced_dataset.csv')

    # Keep only the required columns
    df = df[['title', 'authors', 'tags', 'text', 'url']]

    # Convert to dictionary format
    articles = df.to_dict(orient='records')

    return render_template('main.html', username=session['user'], articles=articles)

@app.route("/article/<int:article_id>")
def article_detail(article_id):
    df = pd.read_csv('../reduced_dataset.csv')
    article = df.iloc[article_id].to_dict()
    
    author = article["authors"]
    author_articles = df[df["authors"] == author]

    # --- Engagement Prediction ---
    
    engagement_model = joblib.load("../models/engagement.pkl")

    inputs = pd.DataFrame([{
        "title": article["title"],
        "tags": ", ".join(article["tags"]),
        "reading_time_min": len(article["text"].split()) // 200
    }])
    predicted_engagement = engagement_model.predict(inputs)[0]

    # --- Feedbacks from MongoDB ---
    feedbacks = db.feedbacks.find({"article_id": article_id})
    
    # Ensure tags are lists before joining
    def clean_tags(raw_tags):
        if isinstance(raw_tags, str):
            try:
                return ", ".join(eval(raw_tags))  # risky if data is not safe
            except:
                return raw_tags
        elif isinstance(raw_tags, list):
            return ", ".join(raw_tags)
        return str(raw_tags)

    
    # --- Author Influence Analysis ---
    author_titles = list(author_articles["title"])
    author_tags = list(author_articles["tags"])
    reading_times = [len(text.split()) // 200 for text in author_articles["text"]]
    input_df = pd.DataFrame({
        "title": author_articles["title"],
        "tags": author_tags,
        "reading_time_min": reading_times
    })
    author_engagements = engagement_model.predict(input_df)

    author_dashboard = {
        "name": author,
        "total_articles": len(author_articles),
        "titles": author_titles,
        "tags": author_tags,
        "engagements": author_engagements.tolist()
    }

    recommendations = list(db.recommendation.find({"article_id": article_id}))

    return render_template(
    "article.html",
    article=article,
    predicted_engagement=predicted_engagement,
    feedbacks=feedbacks,
    article_id=article_id,
    author_dashboard=author_dashboard,
    recommendations=recommendations,  
    zip=zip  # 👈 Add this
     )






@app.route('/search')
def search():
    if 'user' not in session:
        return redirect(url_for('login'))

    query = request.args.get('query', '').lower()
    df = pd.read_csv('../reduced_dataset.csv')
    df = df[df['title'].str.lower().str.contains(query) | df['text'].str.lower().str.contains(query)]
    articles = df.to_dict(orient='records')
    return render_template('main.html', username=session['user'], articles=articles)

@app.route('/advanced-search', methods=['POST'])
def advanced_search():
    if 'user' not in session:
        return redirect(url_for('login'))

    title = request.form['title']
    text = request.form['text']
    input_text = title + ' ' + text

    vector = tfidf_vectorizer.transform([input_text])
    predicted_tags = tag_model.predict(vector)
    tag_list = mlb.inverse_transform(predicted_tags)[0]

    # Filter articles based on predicted tags
    df = pd.read_csv('../reduced_dataset.csv')
    df['tags'] = df['tags'].apply(eval)  # in case it's stored as string
    df = df[df['tags'].apply(lambda tags: any(tag in tags for tag in tag_list))]

    articles = df.to_dict(orient='records')
    return render_template('main.html', username=session['user'], articles=articles, tags=tag_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    entry = db.user_reads.find_one({'user': user})
    urls = entry.get('urls', []) if entry else []

    df = pd.read_csv('../reduced_dataset.csv')
    history_df = df[df['url'].isin(urls)]
    history_df['input_text'] = history_df['title'] + ' ' + history_df['text']
    vectors = tfidf_vectorizer.transform(history_df['input_text'])
    tag_preds = tag_model.predict(vectors)

    tag_list = set()
    for pred in tag_preds:
        tag_list.update(mlb.inverse_transform([pred])[0])

    df['tags'] = df['tags'].apply(eval)
    recommended_df = df[df['tags'].apply(lambda tags: any(tag in tags for tag in tag_list))]
    recommended_df = recommended_df[~recommended_df['url'].isin(urls)]

    articles = recommended_df.to_dict(orient='records')
    return render_template('main.html', username=session['user'], articles=articles, tags=list(tag_list))

    
    
import requests

# Your Groq API key
GROQ_API_KEY = "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"

@app.route("/summarize/<int:article_id>")
def summarize_article(article_id):
    df = pd.read_csv('../reduced_dataset.csv')
    article = df.iloc[article_id].to_dict()
    text = article['text']

    # --- Keyword Extraction ---
    with open("../models/Keyword_Finder_tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = joblib.load(f)

    tfidf_matrix = tfidf_vectorizer.transform([text])
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_keywords = feature_array[tfidf_sorting][:10].tolist()

    # --- Summary via Groq (New model) ---
    groq_summary = "Unable to fetch summary."
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",  # ✅ NEW supported model
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Summarize the article into 3-5 clear bullet points, "
            "avoiding unnecessary introductions. Keep the total under 35 words. Do not include phrases "
            "like 'here are the main points' or 'in summary'."},
                    {"role": "user", "content": f"Summarize this article:\n\n{text[:4000]}"}  # Limit to avoid overrun
                ],
                "temperature": 0.5,
                "max_tokens": 512
            }
        )

        data = response.json()
        if "choices" in data:
            groq_summary = data['choices'][0]['message']['content']
        else:
            groq_summary = f"Groq Error: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        groq_summary = f"Exception: {str(e)}"

    return render_template("summary.html", 
                           title=article['title'], 
                           top_keywords=top_keywords, 
                           groq_summary=groq_summary)


import re
from datetime import datetime
from flask import request, jsonify
import openai  # for Groq compatibility
import os

openai.api_key = os.getenv("GROQ_API_KEY")

# Check for vulgar words (simple example)
def is_vulgar(text):
    vulgar_words = ["badword1", "badword2", "nasty"]  # You can expand this
    return any(word in text.lower() for word in vulgar_words)

# Use Groq API for rephrasing and summarization
import requests

def generate_suggestion(feedback_text):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Clean + format the feedback text prompt
        prompt = f"""
        Rephrase and shorten this user feedback professionally. Remove any vulgar or offensive language.
        Feedback: "{feedback_text}"
        Respond only with the revised text.
        """

        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a content optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()

        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: Unexpected response format from Groq."

    except Exception as e:
        print("Error while generating suggestion:", e)
        return "Error generating suggestion"


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    feedback = data.get("feedback")
    username = data.get("username")
    article_id = data.get("article_id")
    author = data.get("author")
    url = f"/article/{article_id}"

    if is_vulgar(feedback):
        return jsonify({"error": "Vulgar content detected. Please rephrase."}), 400

    suggestion = generate_suggestion(feedback)

    recommendation = {
        "article_id": article_id,
        "author": author,
        "url": url,
        "username": username,
        "original_feedback": feedback,
        "suggestion": suggestion,
        "timestamp": datetime.utcnow().isoformat(),
        "votes": {"up": 0, "down": 0}
    }

    db.recommendation.insert_one(recommendation)
    return jsonify({"message": "Feedback submitted!", "suggestion": suggestion})

@app.route('/vote_feedback/<feedback_id>/<vote_type>', methods=['POST'])
def vote_feedback(feedback_id, vote_type):
    from bson import ObjectId
    if vote_type not in ['up', 'down']:
        return jsonify({"error": "Invalid vote type"}), 400

    db.recommendation.update_one(
        {"_id": ObjectId(feedback_id)},
        {"$inc": {f"votes.{vote_type}": 1}}
    )
    return jsonify({"message": f"{vote_type}voted!"})

    
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
