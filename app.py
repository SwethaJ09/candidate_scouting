import os
from flask import Flask, request, render_template, jsonify
import librosa
import spacy
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import joblib
import speech_recognition as sr
import re
from flask import send_file
import io

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Predefined lists of skills and companies to identify
skills_list = [
    "Spring Boot", "Power BI", "AI", "ML", "Node.js", "React", "SQL", "Java",
    "Project Management", "Data Science", "Web Development", "Cloud Computing",
    "UX Design", "REST APIs", "Android Development", "Data Engineering", "Data Visualization"
]
companies_list = [
    "Amazon", "Microsoft", "Google", "Facebook", "IBM", "Oracle"
]

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Mock emotion model and polarity scores
def extract_features(file_path):
    # Replace with actual MFCC feature extraction
    return np.random.rand(40)

model_path = './random_forest_emotion_polarity_model.pkl'
emotion_model = joblib.load(model_path)

# Helper functions
def evaluate_metrics(speech, polarity_score):
    stop_words = set(nlp.Defaults.stop_words)
    words = speech.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stop_word_count = len(words) - len(filtered_words)
    
    vocab_rating = (len(filtered_words) / len(words)) * 100
    clarity_score = stop_word_count / 100
    speaking_pace = len(filtered_words) / len(words)
    listenability_score = (clarity_score + (polarity_score * 100)) / 2
    return listenability_score, vocab_rating, clarity_score, speaking_pace

def predict_emotion(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract exactly 13 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.T  # Transpose to shape (frames, n_mfcc)

    # Aggregate the MFCCs (e.g., mean) to get a single feature vector
    mfcc_mean = np.mean(mfcc, axis=0).reshape(1, -1)

    # Predict using the Random Forest model
    predicted_emotion = emotion_model.predict(mfcc_mean)[0]
    polarity_score = emotion_model.predict_proba(mfcc_mean).max()

    return predicted_emotion, polarity_score

def extract_audio_metrics(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr  # Total duration in seconds

    # Use np.array_split to avoid ValueError
    segments = np.array_split(y, len(y) // sr)
    pause_duration = sum([1 for segment in segments if np.abs(segment).mean() < 0.01]) / sr

    return duration, pause_duration

# Function to extract entities based on predefined lists
# Function to extract entities based on predefined lists (including PERSON for candidate name)
def extract_entities_custom(doc, skills_list, companies_list):
    found_skills = set()
    found_companies = set()
    candidate_name = None  # To store the extracted name

    # Perform NER on the document
    doc_spacy = nlp(doc)
    
    # Check for PERSON entities (candidate names)
    for ent in doc_spacy.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
    
    # Check for skill matches
    for skill in skills_list:
        if re.search(rf"\b{re.escape(skill)}\b", doc, re.IGNORECASE):
            found_skills.add(skill)

    # Check for company matches
    for company in companies_list:
        if re.search(rf"\b{re.escape(company)}\b", doc, re.IGNORECASE):
            found_companies.add(company)

    return list(found_companies), list(found_skills)

# Function to generate the graph for eligible candidates
import matplotlib.pyplot as plt

def generate_graph(candidates):
    # Generate generic candidate names like 'Candidate 1', 'Candidate 2', etc.
    names = [f"Candidate {i+1}" for i in range(len(candidates))]
    
    eligibility = [1 if candidate["eligibility"] == "Eligible" else 0 for candidate in candidates]  # 1 for eligible, 0 for not
    cgpas = [candidate["cgpa"] for candidate in candidates]
    experiences = [candidate["experience"] for candidate in candidates]
    projects = [candidate["projects"] for candidate in candidates]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for eligibility
    bar_colors = ['green' if eligible else 'red' for eligible in eligibility]
    bars = ax1.bar(names, eligibility, color=bar_colors, alpha=0.6, label='Eligibility')

    # Adding titles and labels for eligibility
    ax1.set_title('Candidate Eligibility Based on Required Skills', fontsize=16)
    ax1.set_xlabel('Candidates', fontsize=14)
    ax1.set_ylabel('Eligibility (1 = Eligible, 0 = Not Eligible)', fontsize=14)
    ax1.set_xticks(range(len(names)))  # Set x-ticks to be the candidates
    ax1.set_xticklabels(names, rotation=45)
    ax1.grid(axis='y')

    # Adding data labels above the bars for eligibility
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, int(yval), ha='center', va='bottom')

    # Create a second y-axis for additional metrics (CGPA, Experience, Projects)
    ax2 = ax1.twinx()

    # Plot CGPA, experience, and projects as line plots
    ax2.plot(names, cgpas, marker='o', color='blue', label='CGPA', linestyle='-', linewidth=2)
    ax2.plot(names, experiences, marker='s', color='orange', label='Experience (Years)', linestyle='--', linewidth=2)
    ax2.plot(names, projects, marker='^', color='purple', label='Projects Completed', linestyle='-.', linewidth=2)

    ax2.set_ylabel('Metrics', fontsize=14)
    ax2.set_ylim(0, max(max(cgpas), max(experiences), max(projects)) + 1)  # Set y-limits for visibility
    ax2.legend(loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.savefig('static/graph.png')  # Save the plot as a .png file
    

# Route for uploading form
@app.route('/')
def upload_form():
    return render_template('index.html')  # Render the HTML template we just created

# Route to process uploaded files
@app.route('/process', methods=['POST'])
def process_files():
    skill = request.form['skill']  # The skill entered by the user in the form
    uploaded_files = request.files.getlist('files')  # List of uploaded files
    candidate_reports = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract audio metrics (duration, pause duration)
        duration, pause_duration = extract_audio_metrics(file_path)

        # Predict emotion and polarity score
        predicted_emotion, polarity_score = predict_emotion(file_path)

        # Transcribe the audio to text using Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            try:
                speech = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                speech = "Could not understand the audio"
            except sr.RequestError as e:
                speech = f"Could not request results from Google Speech Recognition service; {e}"

        # Perform Named Entity Recognition (NER) and extract entities
        companies, skills = extract_entities_custom(speech, skills_list, companies_list)

        # Evaluate the metrics based on the transcribed speech and emotion polarity
        listenability_score, vocab_rating, clarity_score, speaking_pace = evaluate_metrics(speech, polarity_score)

        # Check eligibility: If the input skill is in the extracted skills, mark as eligible
        eligibility = "Eligible" if any(skill.lower() in s.lower() for s in skills) else "Not Eligible"

        # Set placeholder values for CGPA, experience, and projects (you can replace these with actual data if needed)
        cgpa = np.random.uniform(6.5, 9.0)  # Random CGPA between 6.5 and 9.0
        experience = np.random.randint(1, 6)  # Random experience between 1 and 5 years
        projects = np.random.randint(0, 10)  # Random number of projects between 0 and 10

        # Create a report for the candidate
        candidate_reports.append({
            "filename": filename,  # Add candidate name
            "duration": duration,
            "pause_duration": pause_duration,
            "emotion": predicted_emotion,
            "entities": {"companies": companies, "skills": skills},
            "listenability_score": listenability_score,
            "vocab_rating": vocab_rating,
            "clarity_score": clarity_score,
            "speaking_pace": speaking_pace,
            "eligibility": eligibility,  # Add eligibility field
            "cgpa": cgpa,
            "experience": experience,
            "projects": projects
        })

    # Filter eligible candidates
    global eligible_candidates
    eligible_candidates = [r for r in candidate_reports if r["eligibility"] == "Eligible"]

    # Generate graph for listenability scores and metrics
    generate_graph(candidate_reports)

    # Render the report template with the processed data
    return render_template('report.html', skill=skill, reports=candidate_reports, filtered=eligible_candidates)

import io
import random
import matplotlib.pyplot as plt
import numpy as np
from flask import render_template, send_file
import base64

@app.route('/hr_round', methods=['GET', 'POST'])
def hr_round():
    # Eligible candidates (this would ideally come from previous routes)
    
    #eligible_candidates = [
    #    {"name": "Candidate 3", "willingness_to_relocate": 8, "attitude": 9, "problem_solving": 7, "leadership": 6, "behaviors": 8},
    #    {"name": "Candidate 4", "willingness_to_relocate": 7, "attitude": 8, "problem_solving": 9, "leadership": 7, "behaviors": 9},
    #    {"name": "Candidate 5", "willingness_to_relocate": 9, "attitude": 8, "problem_solving": 8, "leadership": 9, "behaviors": 7},
    #    {"name": "Candidate 8", "willingness_to_relocate": 9, "attitude": 9, "problem_solving": 6, "leadership": 8, "behaviors": 7}
    #]

    print(eligible_candidates)
    for candidate in eligible_candidates:
        candidate['name']=candidate['filename'][-5]
        candidate['willingness_to_relocate']= random.randint(1,10)
        candidate['attitude']= random.randint(1,10)
        candidate['problem_solving']= random.randint(1,10)
        candidate['leadership'] = random.randint(1,10)
        candidate['behaviors'] = random.randint(1,10)

    # Step 1: Calculate total score for each candidate
    for candidate in eligible_candidates:
        candidate['total_score'] = (candidate['willingness_to_relocate'] +
                                     candidate['attitude'] +
                                     candidate['problem_solving'] +
                                     candidate['leadership'] +
                                     candidate['behaviors'])

    # Step 2: Sort candidates by total score (descending order)
    ranked_candidates = sorted(eligible_candidates, key=lambda x: x['total_score'], reverse=True)

    # Step 3: Prepare data for bar chart (Rank List)
    names = [candidate['name'] for candidate in ranked_candidates]
    total_scores = [candidate['total_score'] for candidate in ranked_candidates]

    # Create a bar chart for the rank list
    plt.figure(figsize=(10, 6))
    plt.barh(names, total_scores, color='lightblue')
    plt.xlabel('Total Score', fontsize=14)
    plt.title('Rank List of Eligible Candidates for HR Round', fontsize=16)
    plt.xlim(0, max(total_scores) + 5)  # Add some space to the right for readability
    plt.grid(axis='x', linestyle='--')

    # Save the bar chart as a PNG image in memory
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    # Identify the most eligible candidate (the one with the highest total score)
    most_eligible = ranked_candidates[0]  # The first candidate after sorting is the most eligible

    # Step 4: Prepare the radar chart for eligible candidates
    labels = ["Willingness to Relocate", "Attitude", "Problem Solving", "Leadership", "Behaviors"]
    num_vars = len(labels)
    radar_fig, radar_ax = plt.subplots(figsize=(10, 8), dpi=150, subplot_kw=dict(polar=True))

    # Plot radar chart for each candidate
    for candidate in eligible_candidates:
        values = [
            candidate["willingness_to_relocate"],
            candidate["attitude"],
            candidate["problem_solving"],
            candidate["leadership"],
            candidate["behaviors"]
        ]
        values += values[:1]  # Repeat the first value to close the circle
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Repeat the first angle to close the circle

        radar_ax.fill(angles, values, alpha=0.25)
        radar_ax.plot(angles, values, linewidth=2, label=candidate["name"])

    # Add labels and title to the radar chart
    radar_ax.set_xticks(angles[:-1])
    radar_ax.set_xticklabels(labels, fontsize=12)
    radar_ax.set_title('Eligible Candidates for HR Round (Radar Chart)', fontsize=16)
    radar_ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # Save the radar chart as a PNG image in memory
    radar_img = io.BytesIO()
    radar_fig.tight_layout()
    radar_fig.savefig(radar_img, format='png')
    radar_img.seek(0)

    # Convert the radar chart to base64 to display in HTML
    bar_chart_data = base64.b64encode(img.getvalue()).decode('utf-8')
    radar_chart_data = base64.b64encode(radar_img.getvalue()).decode('utf-8')

    # Render the template and pass the images and most eligible candidate data
    return render_template('hr_round.html', 
                           bar_chart_data=bar_chart_data, 
                           radar_chart_data=radar_chart_data, 
                           most_eligible=most_eligible)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

