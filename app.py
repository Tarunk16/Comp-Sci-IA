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
