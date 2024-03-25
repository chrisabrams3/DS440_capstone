#RUN: python fake_news_GUI.py
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog # Added for file upload 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to analyze the text and show the result
def analyze_text(text=None, filename=None):
    if text is None:
        # Get the user input text
        user_input_text = text_entry.get("1.0", 'end-1c')
    else:
        user_input_text = text
    
    # Vectorize the user input text
    user_input_vectorized = vectorizer.transform([user_input_text])
    
    # Make prediction
    prediction = model.predict(user_input_vectorized)
    confidence = model.predict_proba(user_input_vectorized)[0][1]  # Confidence for fake news
    
    # Show result in the result screen
    show_result(prediction, confidence, filename)

# Function to analyze text from file
def analyze_file():
    # Open file dialog to select a .txt file
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        # Read the content of the selected file
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        analyze_text(text, filename)

# Function to show the result in the result screen
def show_result(prediction, confidence, filename=None):
    # Calculate confidence percentage
    confidence_percentage = int(confidence * 100)
    
    if filename:
        file_label.config(text=f"File: {filename}")
    else:
        file_label.config(text="User-Inputed Text")

    # Display prediction and confidence score in the result screen
    prediction_label.config(text=f"Prediction: {'Fake' if prediction == 1 else 'True'}")
    confidence_label.config(text=f"Fake Factor: {confidence_percentage}%")
    
    # Display explanation based on confidence score in the result screen
    if confidence_percentage <= 9:
        explanation = "Highly Reliable - The information seems trustworthy with very low likelihood of being fake news."
    elif confidence_percentage <= 19:
        explanation = "Very Reliable - The content appears to be highly credible with minimal indications of being fake."
    elif confidence_percentage <= 29:
        explanation = "Mostly Reliable - There are slight indications of potential misinformation, but the content is generally trustworthy."
    elif confidence_percentage <= 39:
        explanation = "Moderately Reliable - While some aspects seem credible, there are notable elements suggesting caution."
    elif confidence_percentage <= 49:
        explanation = "Neutral - The reliability of the information is uncertain, requiring further investigation for confirmation."
    elif confidence_percentage <= 59:
        explanation = "Questionable - There are significant doubts about the accuracy of the content, warranting careful scrutiny."
    elif confidence_percentage <= 69:
        explanation = "Likely Unreliable - The information contains substantial inaccuracies and is likely to be misleading."
    elif confidence_percentage <= 79:
        explanation = "Unreliable - There is a high probability that the content is false or misleading, caution is strongly advised."
    elif confidence_percentage <= 89:
        explanation = "Very Unreliable - The information is highly suspect and should be treated as potentially deceptive or false."
    else:
        explanation = "Extremely Unreliable - The content is highly likely to be fake news, and trusting it could lead to misinformation."
    


    
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

# Create "Analyze" button to initiate analysis in the input screen
go_button = tk.Button(input_frame, text="Analyze", command=analyze_text)
go_button.pack(pady=5)

# Create "Or" label in the input screen
or_label = tk.Label(input_frame, text="Or")
or_label.pack()

# Create button to upload text file
upload_button = tk.Button(input_frame, text="Upload Text File", command=analyze_file)
upload_button.pack(pady=5)

# Pack the input frame
input_frame.pack()

# Create frame for result screen
result_frame = tk.Frame(root)

# Create label for filename or "User-Inputed Text" in the result screen
file_label = tk.Label(result_frame, text="")
file_label.pack(pady=10)

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

