from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import json
import os

app = Flask(__name__)

# Load college data from JSON file (or create it if doesn't exist)
def load_college_data():
    if os.path.exists('college_data.json'):
        with open('college_data.json', 'r') as f:
            return json.load(f)
    else:
        # Sample college data with expanded metrics
        colleges = [
            {
                "name": "Harvard University", 
                "avg_gpa": 3.9, 
                "avg_sat": 1500,
                "avg_act": 34,
                "acceptance_rate": 0.05,
                "student_faculty_ratio": 6,
                "median_starting_salary": 75000,
                "avg_financial_aid": 50000,
                "region": "Northeast",
                "size": "Medium",
                "type": "Private",
                "popular_majors": ["Computer Science", "Economics", "Biology"],
                "ranking": 1
            },
            {
                "name": "MIT", 
                "avg_gpa": 3.8, 
                "avg_sat": 1520,
                "avg_act": 35,
                "acceptance_rate": 0.07,
                "student_faculty_ratio": 3,
                "median_starting_salary": 82000,
                "avg_financial_aid": 45000,
                "region": "Northeast",
                "size": "Medium",
                "type": "Private",
                "popular_majors": ["Engineering", "Computer Science", "Mathematics"],
                "ranking": 2
            },
            {
                "name": "UT Austin", 
                "avg_gpa": 3.6, 
                "avg_sat": 1400,
                "avg_act": 30,
                "acceptance_rate": 0.32,
                "student_faculty_ratio": 18,
                "median_starting_salary": 60000,
                "avg_financial_aid": 15000,
                "region": "South",
                "size": "Large",
                "type": "Public",
                "popular_majors": ["Business", "Engineering", "Communication"],
                "ranking": 42
            },
            {
                "name": "UCLA", 
                "avg_gpa": 3.7, 
                "avg_sat": 1380,
                "avg_act": 31,
                "acceptance_rate": 0.14,
                "student_faculty_ratio": 18,
                "median_starting_salary": 63000,
                "avg_financial_aid": 18000,
                "region": "West",
                "size": "Large",
                "type": "Public",
                "popular_majors": ["Psychology", "Biology", "Political Science"],
                "ranking": 20
            },
            {
                "name": "NYU", 
                "avg_gpa": 3.5, 
                "avg_sat": 1300,
                "avg_act": 30,
                "acceptance_rate": 0.21,
                "student_faculty_ratio": 9,
                "median_starting_salary": 65000,
                "avg_financial_aid": 30000,
                "region": "Northeast",
                "size": "Large",
                "type": "Private",
                "popular_majors": ["Business", "Film", "Economics"],
                "ranking": 25
            },
        ]
        with open('college_data.json', 'w') as f:
            json.dump(colleges, f)
        return colleges

colleges = load_college_data()

def predict_acceptance_probability(student_data, college):
    # Define weights for each factor
    weights = {
        "gpa": 0.35,
        "test_scores": 0.25,
        "extracurriculars": 0.15,
        "essay": 0.15,
        "recommendations": 0.05,
        "legacy": 0.03,
        "demonstrated_interest": 0.02
    }
    
    # Calculate test score component (using either SAT or ACT or both)
    test_score = 0
    if student_data.get('sat') and student_data.get('act'):
        # If both are provided, use the one that gives better results
        sat_normalized = (student_data['sat'] / 1600) * 100
        act_normalized = (student_data['act'] / 36) * 100
        test_score = max(sat_normalized, act_normalized)
    elif student_data.get('sat'):
        test_score = (student_data['sat'] / 1600) * 100
    elif student_data.get('act'):
        test_score = (student_data['act'] / 36) * 100
    
    # Normalize GPA (consider weighted vs unweighted)
    if student_data.get('weighted_gpa'):
        # For weighted GPA, consider the scale (usually 5.0)
        normalized_gpa = (student_data['weighted_gpa'] / 5.0) * 100
    else:
        # Regular unweighted GPA on 4.0 scale
        normalized_gpa = (student_data['gpa'] / 4.0) * 100
    
    # Calculate base acceptance score
    score_components = {
        "gpa": normalized_gpa * weights['gpa'],
        "test_scores": test_score * weights['test_scores'],
        "extracurriculars": (student_data['extracurriculars'] / 10) * 100 * weights['extracurriculars'],
        "essay": (student_data['essay'] / 10) * 100 * weights['essay'],
        "recommendations": (student_data.get('recommendations', 7) / 10) * 100 * weights['recommendations'],
        "legacy": (100 if student_data.get('legacy', False) else 0) * weights['legacy'],
        "demonstrated_interest": (student_data.get('demonstrated_interest', 5) / 10) * 100 * weights['demonstrated_interest']
    }
    
    base_score = sum(score_components.values())
    
    # Apply school-specific adjustments
    # 1. Adjust for school selectivity (acceptance rate)
    selectivity_factor = 1 - (college['acceptance_rate'] * 5)  # More selective = higher standard
    
    # 2. Adjust for intended major competitiveness
    major_adjustment = 0
    if student_data.get('intended_major') in college.get('popular_majors', []):
        # Popular majors might be more competitive
        major_adjustment = -5  # Slight penalty for competitive majors
    
    # 3. Geographic diversity adjustment
    geo_adjustment = 0
    if student_data.get('state') and student_data['state'] != college.get('region'):
        # Schools often seek geographic diversity
        geo_adjustment = 5
    
    # Calculate comparison score based on college averages
    avg_gpa_score = (college['avg_gpa'] / 4.0) * 100 * weights['gpa']
    
    # Use higher of SAT/ACT for college average comparison
    if 'avg_sat' in college and 'avg_act' in college:
        avg_test_score = max((college['avg_sat'] / 1600) * 100, (college['avg_act'] / 36) * 100)
    elif 'avg_sat' in college:
        avg_test_score = (college['avg_sat'] / 1600) * 100
    else:
        avg_test_score = (college['avg_act'] / 36) * 100
    
    avg_test_score_component = avg_test_score * weights['test_scores']
    
    college_avg_score = avg_gpa_score + avg_test_score_component
    
    # Calculate final adjusted score
    adjusted_score = base_score + major_adjustment + geo_adjustment
    
    # Convert to probability
    # More selective schools have sharper cutoff curves
    steepness = 1 + selectivity_factor
    midpoint = college_avg_score
    
    # Logistic function for smoother probability curve
    probability = 100 / (1 + np.exp(-steepness * (adjusted_score - midpoint) / 100))
    
    # Cap probability and round
    probability = min(max(probability, 1), 99)  # Never absolutely certain in either direction
    
    # Return probability and component breakdown
    return {
        "probability": round(probability, 1),
        "components": score_components,
        "adjustments": {
            "selectivity": selectivity_factor * -10,  # Convert to a readable penalty value
            "major": major_adjustment,
            "geographic": geo_adjustment
        }
    }

def generate_tips(student_data, results):
    tips = []
    avg_probability = sum(r["probability"] for r in results) / len(results)
    
    # General tips based on application strength
    if avg_probability < 30:
        tips.append("Consider applying to more safety schools with higher acceptance rates.")
        
        # GPA-specific tips
        if student_data['gpa'] < 3.5:
            tips.append("Focus on improving your GPA in your remaining courses.")
        
        # Test score tips
        if student_data.get('sat', 0) < 1300 or student_data.get('act', 0) < 28:
            tips.append("Consider retaking the SAT/ACT to improve your score.")
            
    elif avg_probability < 60:
        tips.append("Your profile has some strengths. Make sure to highlight these in your application.")
        
        # Balanced approach tips
        lowest_component = min(student_data.items(), key=lambda x: x[1] if x[0] in ['gpa', 'extracurriculars', 'essay'] else float('inf'))
        if lowest_component[0] == 'extracurriculars':
            tips.append("Consider taking leadership roles in your extracurricular activities.")
        elif lowest_component[0] == 'essay':
            tips.append("Work on strengthening your personal statement and supplemental essays.")
    else:
        tips.append("Your profile is competitive for most of your selected schools.")
        tips.append("Make sure to submit well-polished applications that highlight your unique qualities.")
    
    # College-specific tips
    colleges_below_50 = [r["college"] for r in results if r["probability"] < 50]
    if colleges_below_50:
        tips.append(f"Your chances at {', '.join(colleges_below_50)} are lower. Consider demonstrating interest through campus visits or reaching out to admissions.")
    
    # Add recommendations for similar colleges
    if avg_probability > 80:
        tips.append("Consider adding more reach schools to your list.")
    elif avg_probability < 40:
        tips.append("Consider adding more safety schools to your list.")
    
    return tips

def find_similar_colleges(student_data, results, all_colleges):
    # Find colleges with similar profiles to those where the student has good chances
    top_matches = [r["college"] for r in results if r["probability"] > 60]
    
    if not top_matches:
        return []
    
    # Find colleges not in the current list with similar profiles
    similar_colleges = []
    for college in all_colleges:
        if college["name"] not in [r["college"] for r in results]:
            # Simple similarity measure based on GPA and SAT
            for top_college_name in top_matches:
                top_college = next(c for c in all_colleges if c["name"] == top_college_name)
                
                gpa_diff = abs(college["avg_gpa"] - top_college["avg_gpa"])
                sat_diff = abs(college["avg_sat"] - top_college["avg_sat"]) / 1600
                
                similarity = 1 - (gpa_diff / 4.0 + sat_diff) / 2
                
                if similarity > 0.8:
                    similar_colleges.append({
                        "name": college["name"],
                        "similarity": round(similarity * 100, 1),
                        "avg_gpa": college["avg_gpa"],
                        "avg_sat": college["avg_sat"]
                    })
                    break
    
    return similar_colleges[:3]  # Return top 3 similar colleges

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect student data
            student_data = {
                "gpa": float(request.form["gpa"]),
                "weighted_gpa": float(request.form.get("weighted_gpa", 0)) or None,
                "sat": int(request.form.get("sat", 0)) or None,
                "act": int(request.form.get("act", 0)) or None,
                "extracurriculars": int(request.form["extracurriculars"]),
                "essay": int(request.form["essay"]),
                "recommendations": int(request.form.get("recommendations", 7)),
                "legacy": request.form.get("legacy") == "yes",
                "demonstrated_interest": int(request.form.get("demonstrated_interest", 5)),
                "intended_major": request.form.get("intended_major", ""),
                "state": request.form.get("state", ""),
                "ethnicity": request.form.get("ethnicity", ""),
                "first_gen": request.form.get("first_gen") == "yes"
            }

            # Input validation
            if not (0 <= student_data["gpa"] <= 4.0):
                raise ValueError("GPA must be between 0.0 and 4.0.")
            if student_data["weighted_gpa"] and not (0 <= student_data["weighted_gpa"] <= 5.0):
                raise ValueError("Weighted GPA must be between 0.0 and 5.0.")
            if student_data["sat"] and not (400 <= student_data["sat"] <= 1600):
                raise ValueError("SAT score must be between 400 and 1600.")
            if student_data["act"] and not (1 <= student_data["act"] <= 36):
                raise ValueError("ACT score must be between 1 and 36.")
            if not (0 <= student_data["extracurriculars"] <= 10):
                raise ValueError("Extracurricular activities must be between 0 and 10.")
            if not (0 <= student_data["essay"] <= 10):
                raise ValueError("Essay quality must be between 0 and 10.")

            # Filter colleges based on region if specified
            filtered_colleges = colleges
            if request.form.get("region"):
                filtered_colleges = [c for c in colleges if c["region"] == request.form.get("region")]
            
            # Calculate results for each college
            results = []
            for college in filtered_colleges:
                result = predict_acceptance_probability(student_data, college)
                results.append({
                    "college": college["name"], 
                    "probability": result["probability"],
                    "components": result["components"],
                    "adjustments": result["adjustments"]
                })
            
            # Sort results by probability
            results.sort(key=lambda x: x["probability"], reverse=True)
            
            # Generate tips
            tips = generate_tips(student_data, results)
            
            # Find similar colleges
            similar_colleges = find_similar_colleges(student_data, results, colleges)

            # Generate Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            college_names = [r["college"] for r in results]
            probabilities = [r["probability"] for r in results]
            
            # Create colormap based on probability
            colors = ['#ff9999' if p < 30 else '#ffcc99' if p < 60 else '#99ff99' for p in probabilities]
            
            bars = ax.bar(college_names, probabilities, color=colors)
            ax.set_ylabel("Acceptance Probability (%)")
            ax.set_title("College Acceptance Chances")
            ax.set_ylim(0, 100)
            
            # Add probability labels on top of bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Convert Plot to Image
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=300)
            buffer.seek(0)
            graph_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Component breakdown chart (radar chart)
            categories = ['GPA', 'Test Scores', 'Extracurriculars', 'Essay', 'Recs']
            if len(results) > 0:
                top_college = results[0]
                values = [
                    top_college["components"]["gpa"] / 0.35, 
                    top_college["components"]["test_scores"] / 0.25,
                    top_college["components"]["extracurriculars"] / 0.15,
                    top_college["components"]["essay"] / 0.15,
                    top_college["components"]["recommendations"] / 0.05
                ]
                
                # Normalize values to 0-100 scale
                values = [min(v, 100) for v in values]
                
                # Create radar chart
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, polar=True)
                
                # Plot the data
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]  # Close the loop
                angles += angles[:1]  # Close the loop
                categories += categories[:1]  # Close the loop
                
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                ax.set_ylim(0, 100)
                ax.set_title(f"Application Component Strength: {top_college['college']}")
                
                # Convert radar chart to image
                radar_buffer = io.BytesIO()
                plt.savefig(radar_buffer, format="png", dpi=300)
                radar_buffer.seek(0)
                radar_graph_url = base64.b64encode(radar_buffer.getvalue()).decode("utf-8")
            else:
                radar_graph_url = None

            # Calculate average probability
            average_probability = sum(r["probability"] for r in results) / len(results) if results else 0

            return render_template(
                "results.html", 
                results=results, 
                graph=graph_url, 
                radar_graph=radar_graph_url,
                average_probability=average_probability, 
                tips=tips,
                similar_colleges=similar_colleges,
                student_data=student_data
            )

        except ValueError as e:
            return render_template("index.html", error=str(e))
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html", regions=list(set(c["region"] for c in colleges)))

@app.route("/add_college", methods=["GET", "POST"])
def add_college():
    if request.method == "POST":
        try:
            new_college = {
                "name": request.form["name"],
                "avg_gpa": float(request.form["avg_gpa"]),
                "avg_sat": int(request.form["avg_sat"]),
                "avg_act": int(request.form["avg_act"]),
                "acceptance_rate": float(request.form["acceptance_rate"]),
                "student_faculty_ratio": int(request.form["student_faculty_ratio"]),
                "median_starting_salary": int(request.form["median_starting_salary"]),
                "avg_financial_aid": int(request.form["avg_financial_aid"]),
                "region": request.form["region"],
                "size": request.form["size"],
                "type": request.form["type"],
                "popular_majors": request.form["popular_majors"].split(","),
                "ranking": int(request.form["ranking"])
            }
            
            # Add to college list and save
            colleges.append(new_college)
            with open('college_data.json', 'w') as f:
                json.dump(colleges, f)
                
            return render_template("add_college.html", message="College added successfully!", regions=list(set(c["region"] for c in colleges)))
        except Exception as e:
            return render_template("add_college.html", error=f"Error adding college: {str(e)}", regions=list(set(c["region"] for c in colleges)))
            
    return render_template("add_college.html", regions=list(set(c["region"] for c in colleges)))

@app.route("/compare", methods=["GET", "POST"])
def compare_colleges():
    if request.method == "POST":
        selected_colleges = request.form.getlist("colleges")
        
        if len(selected_colleges) < 2:
            return render_template("compare.html", colleges=colleges, error="Please select at least two colleges to compare.")
        
        # Get the selected college data
        comparison_data = []
        for college_name in selected_colleges:
            college = next((c for c in colleges if c["name"] == college_name), None)
            if college:
                comparison_data.append(college)
                
        # Generate comparison visualizations
        comparison_metrics = ['avg_gpa', 'avg_sat', 'acceptance_rate', 'student_faculty_ratio', 'median_starting_salary']
        
        # Bar chart comparing key metrics
        fig, axes = plt.subplots(len(comparison_metrics), 1, figsize=(10, 3*len(comparison_metrics)))
        
        for i, metric in enumerate(comparison_metrics):
            ax = axes[i]
            values = [college.get(metric, 0) for college in comparison_data]
            names = [college["name"] for college in comparison_data]
            
            ax.bar(names, values)
            ax.set_title(f"Comparison of {metric.replace('_', ' ').title()}")
            ax.set_xticklabels(names, rotation=45, ha='right')
            
        plt.tight_layout()
        
        # Convert comparison chart to image
        comp_buffer = io.BytesIO()
        plt.savefig(comp_buffer, format="png", dpi=300)
        comp_buffer.seek(0)
        comparison_graph_url = base64.b64encode(comp_buffer.getvalue()).decode("utf-8")
        
        return render_template(
            "compare_results.html",
            colleges=comparison_data,
            graph=comparison_graph_url
        )
        
    return render_template("compare.html", colleges=colleges)

if __name__ == "__main__":
    app.run(debug=True)


