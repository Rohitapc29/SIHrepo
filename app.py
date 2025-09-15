from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # Replace with a secure key for production

# ---------- Skill synonyms and normalizer ----------
skill_synonyms = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'js': 'javascript',
    'py': 'python',
    'reactjs': 'react',
    'nodejs': 'node',
    'ds': 'data science',
    'nlp': 'natural language processing',
    'dl': 'deep learning',
}


def normalize_skills(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().replace(',', ' ')
    for k, v in skill_synonyms.items():
        text = text.replace(k, v)
    return text


# ---------- Load dataset and prepare vectorizer ----------
CSV_PATH = "internshala_data_with_skills.csv"

# Using a placeholder sample DataFrame in case the CSV is missing
SAMPLE_DF = pd.DataFrame([
    {
        'role': 'Data Science Intern', 'company_name': 'Example Analytics', 'hiring_badge': 'Urgent',
        'location': 'Work from home', 'duration': '3 months', 'stipend_perMonth': 15000,
        'skills': 'python machine learning data analysis'
    },
    {
        'role': 'Frontend Intern', 'company_name': 'UI Labs', 'hiring_badge': '',
        'location': 'Bengaluru', 'duration': '2 months', 'stipend_perMonth': 10000,
        'skills': 'javascript react css'
    },
    {
        'role': 'AI Research Intern', 'company_name': 'DeepThink', 'hiring_badge': 'Remote',
        'location': 'Work from home', 'duration': '6 months', 'stipend_perMonth': 25000,
        'skills': 'deep learning python pytorch'
    }
])

try:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        # Fallback to sample data if CSV not found
        df = SAMPLE_DF.copy()

    # Data Cleaning and Preparation
    df['skills'] = df.get('skills', '').fillna('').apply(normalize_skills)
    df['role'] = df.get('role', '').fillna('').str.lower()
    df['company_name'] = df.get('company_name', '').fillna('')
    df['hiring_badge'] = df.get('hiring_badge', '').fillna('')
    df['location'] = df.get('location', '').fillna('')
    df['duration'] = df.get('duration', '').fillna('')
    df['stipend_perMonth'] = df.get('stipend_perMonth', 0).fillna(0)

    # Creating a weighted text representation for each internship
    df['combined_text'] = (
            (df['role'] + " ") * 3 +
            (df['skills'] + " ") * 2 +
            df['company_name'] + " " +
            df['hiring_badge'] + " " +
            df['location'] + " " +
            df['duration'] + " " +
            df['stipend_perMonth'].astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words='english')
    internship_matrix = vectorizer.fit_transform(df['combined_text'])

except Exception as e:
    print("Error loading CSV, falling back to sample data:", e)
    # Re-running preparation logic for the sample data in case of error
    df = SAMPLE_DF.copy()
    df['skills'] = df['skills'].apply(normalize_skills)
    df['role'] = df['role'].str.lower()
    df['combined_text'] = (
            (df['role'] + " ") * 3 +
            (df['skills'] + " ") * 2 +
            df['company_name'] + " " +
            df['hiring_badge'] + " " +
            df['location'] + " " +
            df['duration'] + " " +
            df['stipend_perMonth'].astype(str)
    )
    vectorizer = TfidfVectorizer(stop_words='english')
    internship_matrix = vectorizer.fit_transform(df['combined_text'])


# ---------- Recommendation function (UPDATED) ----------
def recommend_internships(education, skills, interests, location, duration, mode, top_n=6):
    # --- NEW: Filter logic based on Internship Mode ---
    if mode == 'Work from home':
        # Create a boolean mask for Work from home internships
        mask = df['location'] == 'Work from home'
    elif mode == 'Offline':
        # Create a mask for internships that are NOT Work from home
        mask = df['location'] != 'Work from home'
    else:
        # If no mode is specified, don't filter (mask selects all)
        mask = pd.Series([True] * len(df), index=df.index)

    # Apply the mask to both the DataFrame and the similarity matrix
    filtered_df = df[mask]
    filtered_matrix = internship_matrix[mask.to_numpy()]

    # If the filter results in no internships, return an empty list
    if filtered_df.empty:
        return []
    # --- END of new logic ---

    # Normalize user inputs
    skills_input = normalize_skills(skills)
    interests = (interests or '').lower()
    education = (education or '').lower()

    # Create user profile text
    user_profile = (
            f"{education} " +
            f"{(skills_input + ' ') * 2}" +
            f"{(interests + ' ') * 3}" +
            f"{location} {duration}"
    )

    # Calculate similarity against the FILTERED matrix
    user_vec = vectorizer.transform([user_profile])
    similarity_scores = cosine_similarity(user_vec, filtered_matrix).flatten()

    # Get top N indices from the filtered set
    # Ensure top_n is not larger than the number of available internships
    num_recommendations = min(top_n, len(filtered_df))
    top_indices = similarity_scores.argsort()[::-1][:num_recommendations]

    # Get recommended internships from the FILTERED DataFrame
    recommendations = filtered_df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarity_scores[top_indices]
    recommendations['match_percent'] = (recommendations['similarity_score'] * 100).round(1)

    # Ensure all required columns are present
    cols = [
        'role', 'company_name', 'hiring_badge', 'location',
        'duration', 'stipend_perMonth', 'skills', 'match_percent'
    ]
    for c in cols:
        if c not in recommendations.columns:
            recommendations[c] = ''

    return recommendations[cols].to_dict(orient='records')


# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html', logged_in=('user' in session))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        session['user'] = username or 'guest'
        flash('Logged in as ' + session['user'], 'success')
        return redirect(url_for('recommend'))
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        flash('Account created (dummy). Please login.', 'info')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out', 'info')
    return redirect(url_for('index'))


# --- UPDATED /recommend route ---
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        education = request.form.get('education', '')
        skills = request.form.get('skills', '')
        interests = request.form.get('interests', '')
        location = request.form.get('location', '')
        duration = request.form.get('duration', '')
        # Get the new 'Mode' field from the form
        mode = request.form.get('Mode', '')  # e.g., 'Work from home' or 'Offline'
        top_n = int(request.form.get('top_n', 6))

        # Pass the mode to the recommendation function
        recs = recommend_internships(education, skills, interests, location, duration, mode, top_n=top_n)
        return render_template('recommend.html', recommendations=recs, logged_in=('user' in session))

    return render_template('recommend.html', recommendations=None, logged_in=('user' in session))


# --- UPDATED /api/recommend route ---
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json or {}
    # Get the new 'Mode' field from the JSON request body
    mode = data.get('Mode', '')
    recs = recommend_internships(
        data.get('education', ''),
        data.get('skills', ''),
        data.get('interests', ''),
        data.get('location', ''),
        data.get('duration', ''),
        mode,  # Pass the mode to the function
        top_n=int(data.get('top_n', 6))
    )
    return jsonify(recs)


if __name__ == '__main__':
    app.run(debug=True)