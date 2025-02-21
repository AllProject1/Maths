from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fractions import Fraction

app = FastAPI()

# Load dataset
csv_file = "ChatbotDataset.csv"
data = pd.read_csv(csv_file)

questions = data["Question"].tolist()
answers = data["Answer"].tolist()

# Sentence-BERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(questions)

# Directory for images
if not os.path.exists("images"):
    os.makedirs("images")

def plot_fraction_on_number_line(numerator, denominator, filename):
    whole_part = numerator // denominator
    remainder = numerator % denominator
    fraction_value = numerator / denominator

    left = whole_part
    right = whole_part + 1
    step_size = 1 / denominator

    plt.figure(figsize=(12, 2))
    plt.axhline(0, color='black', linewidth=0.8)

    ticks = np.arange(left, right + step_size, step_size)

    for tick in ticks:
        plt.plot(tick, 0, 'ko', markersize=4)
        label = f"{int((tick - whole_part) * denominator)}/{denominator}" if tick != whole_part else f"{int(tick)}"
        plt.text(tick, -0.2, label, ha='center', fontsize=8)

    current_position = whole_part
    for i in range(remainder):
        next_position = whole_part + (i + 1) * step_size
        arrow = FancyArrow(current_position, 0, next_position - current_position, 0,
                           width=0.02, head_width=0.08, head_length=0.05, color='blue')
        plt.gca().add_patch(arrow)
        current_position = next_position

    plt.plot(fraction_value, 0, 'ro', markersize=8)
    plt.text(fraction_value, 0.1, f"{numerator}/{denominator}", ha='center', fontsize=10, color='red')

    plt.xlim(left, right)
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    plt.title(f"Visual Representation of {numerator}/{denominator}")

    plt.savefig(filename)
    plt.close()

@app.get("/ask")
def get_best_answer(query: str):
    if re.match(r'^\d+/\d+$', query):
        numerator, denominator = map(int, query.split('/'))
        filename = f"images/{numerator}_{denominator}.png"
        plot_fraction_on_number_line(numerator, denominator, filename)
        return {"answer": f"Fraction {query} plotted.", "image_url": f"/image/{numerator}_{denominator}"}

    if re.search(r'[\d+\-*/]', query):
        try:
            expression = re.sub(r'(\d+)/(\d+)', r'Fraction(\1, \2)', query)
            result = eval(expression)
            return {"answer": f"The result is {result}"}
        except:
            return {"answer": "Invalid math expression"}

    user_query_embedding = model.encode([query])
    similarities = cosine_similarity(user_query_embedding, question_embeddings).flatten()
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]

    if best_match_score < 0.4:
        return {"answer": "I couldn't understand. Try rephrasing."}

    return {"answer": answers[best_match_index]}

@app.get("/image/{filename}")
def get_image(filename: str):
    file_path = f"images/{filename}.png"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": "Image not found"}, status_code=404)
