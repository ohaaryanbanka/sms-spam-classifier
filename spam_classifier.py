# %% [markdown]
# # SMS Spam Classifier
# A beginner-friendly Machine Learning project that classifies text messages
# as "spam" or "ham" (not spam) using Naive Bayes.
#
# Run this in Google Colab or Jupyter Notebook — each "# %%" marks a new cell.
# In VS Code with the Jupyter extension, these markers also create runnable cells.

# %%
# ---- STEP 1: Import the libraries we need ----
import pandas as pd                                   # for loading & handling the dataset (like Excel, but in code)
from sklearn.model_selection import train_test_split   # to split data into train/test sets
from sklearn.feature_extraction.text import CountVectorizer  # to turn text into numbers
from sklearn.naive_bayes import MultinomialNB          # our classification algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt                        # for plotting the confusion matrix
import seaborn as sns                                   # makes the plot look nicer

# %%
# ---- STEP 2: Load the dataset ----
# This is the classic "SMS Spam Collection" dataset: 5,574 real text messages,
# each labeled "ham" (normal) or "spam".
# Columns are separated by TABS, not commas, so we use sep='\t'.
df = pd.read_csv("sms_spam_dataset.tsv", sep="\t", header=None, names=["label", "message"])

# Peek at the first 5 rows to make sure it loaded correctly
print(df.head())
print("\nTotal messages:", len(df))
print("\nLabel counts:\n", df["label"].value_counts())

# %%
# ---- STEP 3: Convert labels to numbers ----
# Machine learning models work with numbers, not text labels.
# ham -> 0, spam -> 1
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# %%
# ---- STEP 4: Split the data into training and testing sets ----
# We train the model on 80% of the messages, and test it on the remaining 20%
# it has NEVER seen before — this tells us how well it generalizes.
X = df["message"]         # the input: raw text messages
y = df["label_num"]       # the output: 0 or 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # makes the split reproducible (same split every time you run it)
    stratify=y          # keeps the spam/ham ratio consistent in both train and test sets
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

# %%
# ---- STEP 5: Turn text into numbers (vectorization) ----
# ML models can't read words directly. CountVectorizer converts each message
# into a "bag of words" vector — basically a row of word counts.
# stop_words='english' automatically ignores common filler words like "the", "is", "and".
vectorizer = CountVectorizer(stop_words="english")

# fit_transform LEARNS the vocabulary from the training data AND converts it to vectors
X_train_vec = vectorizer.fit_transform(X_train)

# transform (no fit!) applies the SAME vocabulary to the test data
# We never "fit" on test data — that would be leaking information the model shouldn't have.
X_test_vec = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
print("Shape of training matrix:", X_train_vec.shape)

# %%
# ---- STEP 6: Train the model ----
# MultinomialNB = Multinomial Naive Bayes, a fast, classic algorithm for text
# classification. It works especially well with word-count data like ours.
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Model trained!")

# %%
# ---- STEP 7: Evaluate the model ----
# Now we test the model on the unseen test messages and check how accurate it is.
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}\n")

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# %%
# ---- STEP 8: Visualize the confusion matrix ----
# A confusion matrix shows exactly where the model got confused:
# how many hams were correctly called ham, how many spams were missed, etc.
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# %%
# ---- STEP 9: Try it on your own messages! ----
def predict_message(text):
    """Takes a raw text message and returns 'spam' or 'ham' with a confidence score."""
    vec = vectorizer.transform([text])          # convert the single message to the same vector format
    prediction = model.predict(vec)[0]           # 0 or 1
    probability = model.predict_proba(vec)[0]    # [P(ham), P(spam)]
    label = "spam" if prediction == 1 else "ham"
    confidence = probability[prediction]
    return label, confidence

test_messages = [
    "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!!!",
    "Hey, are we still on for the gym session at 6pm today?",
    "URGENT: Your account has been suspended. Verify your details immediately.",
    "Bro don't forget to bring the charger tomorrow"
]

for msg in test_messages:
    label, confidence = predict_message(msg)
    print(f"[{label.upper():5s} | {confidence:.1%}] {msg}")
