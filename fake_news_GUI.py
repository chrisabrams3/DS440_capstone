#RUN: python fake_news_GUI.py
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to analyze the text and show the result
def analyze_text():
    # Get the user input text
    user_input_text = text_entry.get("1.0", 'end-1c')
    
    # Vectorize the user input text
    user_input_vectorized = vectorizer.transform([user_input_text])
    
    # Make prediction
    prediction = model.predict(user_input_vectorized)
    confidence = model.predict_proba(user_input_vectorized)[0][1]  # Confidence for fake news
    
    # Show result in the result screen
    show_result(prediction, confidence)

# Function to show the result in the result screen
def show_result(prediction, confidence):
    # Calculate confidence percentage
    confidence_percentage = int(confidence * 100)
    
    # Display prediction and confidence score in the result screen
    prediction_label.config(text=f"Prediction: {'Fake' if prediction == 1 else 'True'}")
    confidence_label.config(text=f"Confidence Score: {confidence_percentage}%")
    
    # Display explanation based on confidence score in the result screen
    if confidence_percentage <= 19:
        explanation = "Explanation for confidence score 0-19%"
    elif confidence_percentage <= 39:
        explanation = "Explanation for confidence score 20-39%"
    elif confidence_percentage <= 59:
        explanation = "Explanation for confidence score 40-59%"
    elif confidence_percentage <= 79:
        explanation = "Explanation for confidence score 60-79%"
    else:
        explanation = "Explanation for confidence score 80-100%"
    
    explanation_label.config(text=explanation)
    
    # Hide input screen and show result screen
    input_frame.pack_forget()
    result_frame.pack()

# Function to go back to the input screen
def go_back():
    # Hide result screen and show input screen
    result_frame.pack_forget()
    input_frame.pack()

# Create the main window
root = tk.Tk()
root.title("Fake News Detector")

# Create frame for input screen
input_frame = tk.Frame(root)

# Create title label in the input screen
title_label = tk.Label(input_frame, text="Fake News Detector", font=("Helvetica", 16))
title_label.pack(pady=10)

# Create text entry box in the input screen
text_entry = tk.Text(input_frame, height=10, width=50)
text_entry.pack(pady=10)

# Create "GO" button to initiate analysis in the input screen
go_button = tk.Button(input_frame, text="GO", command=analyze_text)
go_button.pack(pady=5)

# Pack the input frame
input_frame.pack()

# Create frame for result screen
result_frame = tk.Frame(root)

# Create labels for prediction, confidence score, and explanation in the result screen
prediction_label = tk.Label(result_frame, text="")
prediction_label.pack(pady=10)

confidence_label = tk.Label(result_frame, text="")
confidence_label.pack(pady=5)

explanation_label = tk.Label(result_frame, text="")
explanation_label.pack(pady=10)

# Create "Home" button to go back to the input screen in the result screen
home_button = tk.Button(result_frame, text="Home", command=go_back)
home_button.pack(pady=10)

# Hide the result frame initially
result_frame.pack_forget()

root.mainloop()
