from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from ml_model import predict_acceptance_probability

app = Flask(__name__)

# Sample college data
colleges = [
    {"name": "Harvard University", "avg_gpa": 3.9, "avg_sat": 1500},
    {"name": "MIT", "avg_gpa": 3.8, "avg_sat": 1520},
    {"name": "UT Austin", "avg_gpa": 3.6, "avg_sat": 1400},
    {"name": "UCLA", "avg_gpa": 3.7, "avg_sat": 1380},
    {"name": "NYU", "avg_gpa": 3.5, "avg_sat": 1300},
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gpa = float(request.form["gpa"])
        sat = int(request.form["sat"])
        extracurriculars = int(request.form["extracurriculars"])
        essay = int(request.form["essay"])

        results = []
        for college in colleges:
            probability = predict_acceptance_probability(gpa, sat, extracurriculars, essay, college["avg_gpa"], college["avg_sat"])
            results.append({"college": college["name"], "probability": probability})

        # Generate Bar Chart
        fig, ax = plt.subplots()
        college_names = [r["college"] for r in results]
        probabilities = [r["probability"] for r in results]
        ax.bar(college_names, probabilities)
        ax.set_ylabel("Acceptance Probability (%)")
        ax.set_title("College Acceptance Chances")
        plt.xticks(rotation=45)

        # Convert Plot to Image
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        graph_url = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return render_template("results.html", results=results, graph=graph_url)

    return render_template("index.html")
    

if __name__ == "__main__":
    app.run(debug=True)

def predict_acceptance_probability(gpa, sat, extracurriculars, essay, avg_gpa, avg_sat):
    # Define weights for each factor
    gpa_weight = 0.4
    sat_weight = 0.4
    extracurricular_weight = 0.1
    essay_weight = 0.1
    
    # Normalize scores
    normalized_gpa = (gpa / 4.0) * 100  # Convert GPA to a percentage
    normalized_sat = (sat / 1600) * 100  # Convert SAT to a percentage
    
    # Calculate the weighted score
    weighted_score = (normalized_gpa * gpa_weight +
                      normalized_sat * sat_weight +
                      (extracurriculars * 10) * extracurricular_weight +  # Assume a scale of 0-10 for extracurriculars
                      (essay * 10) * essay_weight)  # Assume a scale of 0-10 for essay quality
    
    # Compare with average college data
    avg_score = (avg_gpa / 4.0) * 100 * gpa_weight + (avg_sat / 1600) * 100 * sat_weight
    
    # Calculate probability of acceptance
    acceptance_probability = min(weighted_score / avg_score * 100, 100)  # Cap at 100%
    
    return round(acceptance_probability, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            gpa = float(request.form["gpa"])
            sat = int(request.form["sat"])
            extracurriculars = int(request.form["extracurriculars"])
            essay = int(request.form["essay"])

            # Input validation
            if not (0 <= gpa <= 4.0):
                raise ValueError("GPA must be between 0.0 and 4.0.")
            if not (0 <= sat <= 1600):
                raise ValueError("SAT score must be between 0 and 1600.")
            if not (0 <= extracurriculars <= 10):
                raise ValueError("Extracurricular activities must be between 0 and 10.")
            if not (0 <= essay <= 10):
                raise ValueError("Essay quality must be between 0 and 10.")

            results = []
            for college in colleges:
                probability = predict_acceptance_probability(gpa, sat, extracurriculars, essay, college["avg_gpa"], college["avg_sat"])
                results.append({"college": college["name"], "probability": probability})

            # Generate Bar Chart
            fig, ax = plt.subplots()
            college_names = [r["college"] for r in results]
            probabilities = [r["probability"] for r in results]
            ax.bar(college_names, probabilities)
            ax.set_ylabel("Acceptance Probability (%)")
            ax.set_title("College Acceptance Chances")
            plt.xticks(rotation=45)

            # Convert Plot to Image
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            graph_url = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Calculate average probability
            average_probability = sum(r["probability"] for r in results) / len(results)
            tips = generate_tips(average_probability)

            return render_template("results.html", results=results, graph=graph_url, average_probability=average_probability, tips=tips)

        except ValueError as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


