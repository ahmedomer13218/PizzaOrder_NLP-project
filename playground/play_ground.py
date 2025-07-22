import tensorflow as tf
import contractions
from spellchecker import SpellChecker
from transformers import BertTokenizer
import spacy
import json
import numpy as np
import re
import pandas as pd
from tqdm import tqdm

spell = SpellChecker()

punctuation = re.compile(r"[^\w\s]")

data = []
with open('PIZZA_train.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))


def extract_text(data):
    return data['train.SRC'], data['train.EXR'], data['train.TOP'], data['train.TOP-DECOUPLED']


sentences, explanations, topics, decoupled_topics = map(
    np.array, zip(*map(extract_text, data)))

def preprocess_text(text):

    text = contractions.fix(text)  # expand contractions

    words = text.split()
    corrected_words = [spell.correction(word) or word for word in words]
    text = ' '.join(corrected_words)

    text = tf.strings.strip(text)  # leading/trailing whitespace removal

    text = tf.strings.lower(text)

    text = punctuation.sub("", text.numpy().decode(
        'utf-8'))  # punctuation removal


    return text


pd_sentences = pd.Series(sentences)

tqdm.pandas(desc="Processing Data")

processed_pd_sentences = pd_sentences.progress_apply(preprocess_text)
