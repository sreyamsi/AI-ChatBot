from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')

# Load required files
lemmatizer = WordNetLemmatizer()
keys = json.load(open('keys.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'key': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, keys_json):
    tag = ints[0]['key']
    list_of_keys = keys_json['keys']
    for i in list_of_keys:
        if i['tag'] == tag:
            return random.choice(i['responses'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    message = request.form["msg"]

    # Check for stop command
    if message.lower() in ["exit", "stop", "quit"]:
        return jsonify({"response": "Chat ended. Thank you!"})

    ints = predict_class(message)
    res = get_response(ints, keys)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)


