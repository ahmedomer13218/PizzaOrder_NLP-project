from transformers import BertTokenizer

from spellchecker import SpellChecker


import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import re
import json
import streamlit as st


model1 = load_model('models/shared_encoder_decoder02.keras')
model2 = load_model('models/shared_encoder_decoder2.keras')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


entity_labels_drink = np.load('data/processed/entity_labels_drink.npy')
entity_labels_pizza = np.load('data/processed/entity_labels_pizza.npy')
enitity_labels_second = np.load('data/processed/entity_labels_second.npy')

entities_id_drink = {i+1: str(e) for i, e in enumerate(entity_labels_drink)}
entities_id_drink[0] = 'O'
entities_id_drink


entities_id_pizza = {i+1: str(e) for i, e in enumerate(entity_labels_pizza)}
entities_id_pizza[0] = 'O'
entities_id_pizza


enitities_id_second = {i+1: str(e)
                       for i, e in enumerate(enitity_labels_second)}
enitities_id_second[0] = 'O'
enitities_id_second

spell = SpellChecker()


def spell_check(sentence):
    words = sentence.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return ' '.join(corrected_words)

def process_second_entities(entities_second):
    current_entity = 'O'
    processed_entities = []
    for entity in entities_second:
        if entity == 'O' and current_entity == 'O':
            processed_entities.append('O')
        elif entity == 'O' and current_entity != 'O':
            if current_entity.startswith('B-'):
                current_entity = 'I-'+current_entity[2:]
            processed_entities.append(current_entity)
        elif entity.startswith('I-') and current_entity == 'O':
            current_entity = 'B-'+entity[2:]
            processed_entities.append(current_entity)
        else:
            current_entity = entity
            processed_entities.append(current_entity)

    return processed_entities


def get_TOP(words, predictions1, predictions2, predictions3):
    result = "(ORDER "
    i = 0
    not_flag = False
    c = min(len(words), len(predictions1), len(
        predictions2), len(predictions3))
    while i < c:
        if i < c and predictions3[i].startswith('B-'):
            tag = predictions3[i][2:]
            result += f"({tag} "
            while i < c and (predictions3[i].startswith('B-') or predictions3[i].startswith('I-')):
                if predictions1[i].startswith('B-'):
                    if (predictions1[i].startswith('B-NOT_')):
                        result += f"(NOT ({predictions1[i][6:]} {words[i]} ) "
                        not_flag = True
                    else:
                        result += f"({predictions1[i][2:]} {words[i]} ) "
                elif predictions2[i].startswith('B-'):
                    # Handle multi-word drink types
                    if i+1 < c and predictions2[i+1].startswith('I-'):
                        result += f"({predictions2[i][2:]} {words[i]} {words[i+1]} ) "
                        i += 1
                    else:
                        result += f"({predictions2[i][2:]} {words[i]} ) "
                else:
                    result += f"{words[i]} "
                i += 1
                if i < c and predictions3[i] == 'O':
                    break
            if not_flag:
                result = result.rstrip() + " ) ) "
                not_flag = False
            else:
                result = result.rstrip() + " ) "
        else:
            result += f"{words[i]} "
            i += 1
    result = result.rstrip() + " )"
    return result


def get_prediction_entities1(processed_sentences, predictions, entities_id):
    pred_entities = []
    for i in range(predictions.shape[0]):
        sen = [entities_id[np.argmax(predictions[i][j])]
               for j in range(predictions.shape[1])]
        pred_entities.append(sen[1:len(processed_sentences)+1])
    return pred_entities


def get_prediction(sentence, model1, model2):
    print(sentence)
    sentence = sentence.split()
    encoded_input = tokenizer(sentence,
                              truncation=True,
                              padding="max_length",
                              max_length=30,
                              is_split_into_words=True)

    input_ids = np.array([encoded_input["input_ids"]])
    raw_predictions1 = np.array(model1.predict(input_ids))
    raw_predictions2 = np.array(model2.predict(input_ids))

    entities_pizza = np.array(get_prediction_entities1(
        sentence, raw_predictions1[0], entities_id_pizza), dtype=object)
    entities_drink = np.array(get_prediction_entities1(
        sentence, raw_predictions1[1], entities_id_drink), dtype=object)
    entities_second = np.array(get_prediction_entities1(
        sentence, raw_predictions2, enitities_id_second), dtype=object)
    return sentence, entities_pizza[0], entities_drink[0], process_second_entities(entities_second[0])


def predict(sentence):
    ws, p1, p2, p3 = get_prediction(sentence, model1, model2)
    st.write(ws, p1, p2, p3)
    return get_TOP(ws, p1, p2, p3)



def tokenize(s):
    tokens = re.findall(r'\(|\)|[^\s()]+', s)
    return tokens


def parse_tokens(tokens):
    # Parse tokens into a nested list structure
    stack = []
    current_list = []
    for token in tokens:
        if token == '(':
            stack.append(current_list)
            current_list = []
        elif token == ')':
            finished = current_list
            current_list = stack.pop()
            current_list.append(finished)
        else:
            current_list.append(token)
    return current_list


def normalize_structure(tree):
    if not isinstance(tree, list):
        return None

    def is_key(token):
        return token in [
            "ORDER", "PIZZAORDER", "DRINKORDER", "NUMBER", "SIZE", "STYLE", "TOPPING",
            "COMPLEX_TOPPING", "QUANTITY", "VOLUME", "DRINKTYPE", "CONTAINERTYPE", "NOT"
        ]

    # Clean the list by keeping sublists and tokens as-is for further analysis
    cleaned = []
    for el in tree:
        cleaned.append(el)

    if len(cleaned) > 0 and isinstance(cleaned[0], str) and is_key(cleaned[0]):
        key = cleaned[0]
        if key == "ORDER":
            pizzaorders = []
            drinkorders = []
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    if "PIZZAORDER" in node:
                        if isinstance(node["PIZZAORDER"], list):
                            pizzaorders.extend(node["PIZZAORDER"])
                        else:
                            pizzaorders.append(node["PIZZAORDER"])
                    if "DRINKORDER" in node:
                        if isinstance(node["DRINKORDER"], list):
                            drinkorders.extend(node["DRINKORDER"])
                        else:
                            drinkorders.append(node["DRINKORDER"])
                    if node.get("TYPE") == "PIZZAORDER":
                        pizzaorders.append(node)
                    if node.get("TYPE") == "DRINKORDER":
                        drinkorders.append(node)
            result = {}
            if pizzaorders:
                result["PIZZAORDER"] = pizzaorders
            if drinkorders:
                result["DRINKORDER"] = drinkorders
            if result:
                return {"ORDER": result}
            else:
                return {}

        elif key == "PIZZAORDER":
            number = None
            size = None
            style = None
            toppings = []
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "NUMBER":
                        number = node["VALUE"]
                    elif t == "SIZE":
                        size = node["VALUE"]
                    elif t == "STYLE":
                        style = node["VALUE"]
                    elif t == "TOPPING":
                        toppings.append(node)
            result = {}
            if number is not None:
                result["NUMBER"] = number
            if size is not None:
                result["SIZE"] = size
            if style is not None:
                result["STYLE"] = style
            if toppings:
                result["AllTopping"] = toppings
            # Mark type internally, will remove later
            result["TYPE"] = "PIZZAORDER"
            return result

        elif key == "DRINKORDER":
            number = None
            volume = None
            drinktype = None
            containertype = None
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "NUMBER":
                        number = node["VALUE"]
                    elif t == "VOLUME" or t == "SIZE":
                        volume = node["VALUE"]
                    elif t == "DRINKTYPE":
                        drinktype = node["VALUE"]
                    elif t == "CONTAINERTYPE":
                        containertype = node["VALUE"]
            result = {}
            if number is not None:
                result["NUMBER"] = number
            if volume is not None:
                result["SIZE"] = volume
            if drinktype is not None:
                result["DRINKTYPE"] = drinktype
            if containertype is not None:
                result["CONTAINERTYPE"] = containertype
            result["TYPE"] = "DRINKORDER"
            return result

        elif key in ["NUMBER", "SIZE", "STYLE", "VOLUME", "DRINKTYPE", "CONTAINERTYPE", "QUANTITY"]:
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            value_str = " ".join(values).strip()
            return {
                "TYPE": key,
                "VALUE": value_str
            }

        elif key == "TOPPING":
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            topping_str = " ".join(values).strip()
            return {
                "TYPE": "TOPPING",
                "NOT": False,
                "Quantity": None,
                "Topping": topping_str
            }

        elif key == "COMPLEX_TOPPING":
            quantity = None
            topping = None
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "QUANTITY":
                        quantity = node["VALUE"]
                    elif t == "TOPPING":
                        topping = node["Topping"]
            return {
                "TYPE": "TOPPING",
                "NOT": False,
                "Quantity": quantity,
                "Topping": topping
            }

        elif key == "NOT":
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict) and node.get("TYPE") == "TOPPING":
                    node["NOT"] = True
                    if "Quantity" not in node:
                        node["Quantity"] = None
                    return node
            return None

    else:
        # Try to parse sublists and combine orders found
        combined_order = {"PIZZAORDER": [], "DRINKORDER": []}
        found_order = False

        for el in cleaned:
            node = normalize_structure(el)
            if isinstance(node, dict):
                if "ORDER" in node:
                    found_order = True
                    order_node = node["ORDER"]
                    if "PIZZAORDER" in order_node:
                        combined_order["PIZZAORDER"].extend(
                            order_node["PIZZAORDER"])
                    if "DRINKORDER" in order_node:
                        combined_order["DRINKORDER"].extend(
                            order_node["DRINKORDER"])
                elif node.get("TYPE") == "PIZZAORDER":
                    found_order = True
                    combined_order["PIZZAORDER"].append(node)
                elif node.get("TYPE") == "DRINKORDER":
                    found_order = True
                    combined_order["DRINKORDER"].append(node)

        if found_order:
            final = {}
            if combined_order["PIZZAORDER"]:
                final["PIZZAORDER"] = combined_order["PIZZAORDER"]
            if combined_order["DRINKORDER"]:
                final["DRINKORDER"] = combined_order["DRINKORDER"]
            return {"ORDER": final} if final else {}

        return None


def remove_type_keys(obj):
    # Recursively remove "TYPE" keys from all dictionaries
    if isinstance(obj, dict):
        obj.pop("TYPE", None)
        for k, v in obj.items():
            remove_type_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_type_keys(item)


def preprocess(text):
    tokens = tokenize(text)
    parsed = parse_tokens(tokens)
    result = normalize_structure(parsed)
    remove_type_keys(result)
    return result


# input_str = input("Enter your order: ")
# # input_str = "(ORDER i need (PIZZAORDER (NUMBER a ) (SIZE medium ) (TOPPING ham ) and (TOPPING pineapple ) pizza ) and (DRINKORDER (NUMBER a ) (SIZE small ) (DRINKTYPE iced tea ) ) )"

# result = preprocess(predict(input_str))

# print(json.dumps(result, indent=2))


st.title("Order Processing App")

input_str = st.text_input("Enter your order:")

if input_str:
    r = predict(spell_check(input_str))
    result = preprocess(r)
    st.write(r)
    st.json(result)