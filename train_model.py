import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from pathlib import Path

# Ensure nltk data is available
nltk.download("punkt")
nltk.download("wordnet")

# Define base and data paths
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
MODEL_PATH = BASE / "chatbot_model.pkl"
INTENTS_PATH = DATA_DIR / "intent.json"

# Initialize tools
lemmatizer = WordNetLemmatizer()

# Load intents
with open(INTENTS_PATH, "r") as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Tokenize and lemmatize patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model and preprocessing data
model.save(BASE / "chatbot_model.h5", hist)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"words": words, "classes": classes, "intents": intents}, f)

print("✅ Model training complete and saved!")
print(f"📁 Model saved to: {MODEL_PATH}")
