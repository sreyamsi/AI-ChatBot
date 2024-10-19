import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer and load model
lemmatizer = WordNetLemmatizer()
keys = json.load(open('keys.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

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
    ERROR_THRESHOLD = 0.15
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'key': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def get_response(keys_list, keys_json):
    if not keys_list:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

    tag = keys_list[0]['key']
    list_of_keys = keys_json['keys']

    for i in list_of_keys:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

# Chat loop
while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, keys)
    print(f"Chatbot: {res}")
