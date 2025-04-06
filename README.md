# 🧠 Smart_Read — AI-Powered Content Analyzer

> Harnessing AI to understand content, predict engagement, and personalize experiences.

## 📌 Overview

Smart_Read is a full-stack, AI-driven application developed during the **Coderelate Hackathon**. It uses NLP and ML to analyze digital articles and deliver:

- ✅ **Tag Prediction**
- ✅ **Engagement Level Forecasting**
- ✅ **Keyword Extraction**
- ✅ **Author Influence Analytics**
- ✅ **Personalized Recommendations**

[🔗 GitHub Repository](https://github.com/Sri-Krishnan007/Smart_Read)

---

## 🚀 Features

### 🏷️ Tag Modeling
> Multi-label classification using TF-IDF + Logistic Regression (OneVsRest)

### 📈 Engagement Prediction
> Classification of low/medium/high engagement based on engineered metrics  
Model: Random Forest | Accuracy: ~81%

### 🔑 Keyword Extraction
> Extract top-N relevant terms for summaries/snippets via TF-IDF

### 👤 Author Analytics
> Aggregated engagement by author with visual feedback

### 🧠 AI-Assisted Content Feedback
> Uses Groq API to offer improvement suggestions for authors

---

## 🧪 Round 1 — Machine Learning R&D

### 🔍 Problem Definition

We addressed 3 key tasks:

1. **Tag Modeling**  
   Predict tags based on the article's title and content (multi-label classification)

2. **Engagement Prediction**  
   Predict popularity level based on metadata (regression → converted to classification)

3. **Keyword Extraction**  
   Identify top relevant terms for summarization and discoverability

---

### 🧹 Data Preprocessing

- Removed nulls, duplicates  
- Text cleaning pipeline:
  - Lowercasing  
  - Punctuation removal  
  - Stopword removal  
  - Lemmatization  
- Cleaned metadata (authors, URLs, timestamps)  
- Outlier removal based on engagement score  

---

### 📊 Exploratory Data Analysis (EDA)

- Tag frequency distribution  
- Article text statistics:
  - Word count  
  - Sentence length  
  - Reading time (based on avg 200 wpm)  
- Engagement vs features correlation  
- Title sentiment polarity and subjectivity  
- TF-IDF vector visualizations using WordClouds  

---

### 🧠 Feature Engineering

Generated custom features:
- `word_count`, `title_length`, `text_length`  
- `reading_time_min` (estimated)  
- `title_sentiment`, `text_sentiment`  
- `polarity`, `subjectivity`  
- TF-IDF features from `text_combined`  

---

### 📈 Modeling (Prototype Stage)

#### ✅ Engagement Prediction

```python
df['engagement_score'] = (
    df['word_count'] * 0.3 +
    df['reading_time_min'] * 0.2 +
    df['title_length'] * 0.2 +
    df['text_length'] * 0.3
)
df['engagement_level'] = pd.qcut(df['engagement_score'], q=3, labels=[0, 1, 2])
```

- Model: Random Forest Classifier  
- Labels: Low (0), Medium (1), High (2)  
- Accuracy: ~81%

#### ✅ Tag Modeling

- Text Features: TF-IDF (title + content)  
- Model: OneVsRestClassifier (Logistic Regression)  
- Output: One or more predicted tags per article  
- Encoding: MultiLabelBinarizer  

#### ✅ Keyword Extraction

- Used `TFIDFVectorizer`  
- Selected top-N scoring terms  
- Used for summarization/snippet generation  

---

## 🚀 Round 2 — Full-Stack Implementation

### 💻 Tech Stack

- Frontend: HTML, CSS, JS  
- Backend: Flask  
- Database: MongoDB  
- ML Models:
  - `tag_model.pkl`
  - `engagement.pkl`
  - `Keyword_Finder_tfidf_vectorizer.pkl`
  - `tfidf_vectorizer.pkl`
  - `multilabel_binarizer.pkl`
- AI API: Groq (for content improvement feedback)

---

### 🔐 Authentication System

- Flask-based login/signup  
- User credentials securely stored in MongoDB  
- Post-login redirect to `main.html`

---

### 🗂️ Main Page (main.html)

- Loads data from `final_data.csv`
- Displays article cards with:
  - Title, tags, authors, reading time
- Clickable cards redirect to detailed article view

---

### 🔎 Search Functionalities

- **Simple Search:** Based on keywords in title or content  
- **Advanced Search:** 
  - Predicts tags based on input  
  - Filters articles by relevant tags

---

### 📄 Article Detail Page

- Shows full content, author, timestamp, etc.  
- Allows user feedback submission  
- Feedback analyzed via sentiment analysis  
- Stored with associated sentiment

---

### 📈 Engagement Prediction

- Predicts article engagement level using `engagement.pkl`  
- Uses engineered features like reading time and title richness

---

### 🧠 Keyword-Based Summarization

- Extracts top keywords using:
  - `Keyword_Finder_tfidf_vectorizer.pkl`
- If URL is present:
  - Content scraped and summarized

---

### 👤 Author Influence Analysis

- Aggregates engagement score by author  
- Visual or tabular representation of author impact

---

### 🧑‍💻 Personalized Recommendations

- Clicked articles are stored per user in MongoDB  
- Related articles recommended based on tag similarity

---

### 💬 Content Optimization Assistant (Groq API)

- Authors receive feedback to improve articles  
- Suggestions stored in recommendation collection:
  - author, url, suggestion

---

## 🧭 Future Enhancements

- Use BERT/Transformer models for tag prediction  
- Author trend visualization via interactive plots  
- Add collaborative filtering for better personalization  
- Convert to a Progressive Web App (PWA)

---

## 📦 Installation

```bash
git clone https://github.com/Sri-Krishnan007/Smart_Read.git
cd Smart_Read
pip install -r requirements.txt
python app.py
```

---

## 🏁 Conclusion

Smart_Read combines the power of AI and NLP to provide:

- Intelligent article analysis  
- Engagement prediction  
- Author impact analytics  
- Personalized content discovery  
- Feedback-driven content improvement
