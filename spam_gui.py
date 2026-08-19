"""
SMS Spam Classifier - Desktop GUI version (Tkinter)
Run this with: python spam_gui.py
Make sure sms_spam_dataset.tsv is in the same folder as this file.
"""

import tkinter as tk
from tkinter import font as tkfont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---- 1. Train the model (same steps as the notebook, just running once at startup) ----
print("Loading dataset and training model, please wait...")

df = pd.read_csv("sms_spam_dataset.tsv", sep="\t", header=None, names=["label", "message"])
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Model ready! Opening window...")


def classify_message():
    """Runs when the 'Check Message' button is clicked."""
    text = message_entry.get("1.0", tk.END).strip()

    if not text:
        result_label.config(text="Type a message first!", fg="#888888")
        return

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = probability[prediction]

    if prediction == 1:
        result_label.config(text=f"SPAM  ({confidence:.0%} confidence)", fg="#d9534f")
    else:
        result_label.config(text=f"HAM  ({confidence:.0%} confidence)", fg="#2e7d32")


def clear_message():
    message_entry.delete("1.0", tk.END)
    result_label.config(text="", fg="#000000")


# ---- 2. Build the window ----
window = tk.Tk()  # tk.Tk() creates the main application window
window.title("SMS Spam Classifier")
window.geometry("480x380")
window.resizable(False, False)

heading_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
normal_font = tkfont.Font(family="Helvetica", size=11)
result_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

tk.Label(window, text="SMS Spam Classifier", font=heading_font).pack(pady=(20, 5))
tk.Label(window, text="Type or paste a message below:", font=normal_font).pack()

message_entry = tk.Text(window, height=6, width=48, font=normal_font, wrap="word")
message_entry.pack(pady=10, padx=20)

button_frame = tk.Frame(window)
button_frame.pack(pady=5)

check_button = tk.Button(button_frame, text="Check Message", font=normal_font,
                          command=classify_message, bg="#4a90d9", fg="white", padx=12, pady=6)
check_button.grid(row=0, column=0, padx=5)

clear_button = tk.Button(button_frame, text="Clear", font=normal_font,
                          command=clear_message, padx=12, pady=6)
clear_button.grid(row=0, column=1, padx=5)

result_label = tk.Label(window, text="", font=result_font)
result_label.pack(pady=20)

# ---- 3. Start the app ----
window.mainloop()
