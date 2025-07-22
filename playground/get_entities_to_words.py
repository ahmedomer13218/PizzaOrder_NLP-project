import json
import numpy as np
from tqdm import tqdm
import re

# data = []
# with open('PIZZA_train.json', 'r') as f:
#     for line in f:
#         data.append(json.loads(line))

with open('PIZZA_dev.json', 'r') as file:
    data = json.load(file)


# def extract_text(data):
#     return data['train.SRC'], data['train.EXR'], data['train.TOP'], data['train.TOP-DECOUPLED']

def extract_text(data):
    return data['dev.SRC'], data['dev.EXR'], data['dev.TOP'], data['dev.PCFG_ERR']

sentences, explanations, topics, decoupled_topics = map(
    np.array, zip(*map(extract_text, data)))

entities = set([word[1:]
               for t in topics for word in t.split() if word.isupper()])

enitities_exclude_not = entities - {'NOT'}

negate_mapping = {}
for entity in entities:
    if entity !='NOT':
        negate_mapping[entity] = "NOT_"+entity

def negate_topic(topic):
    not_array = topic.split('NOT')
    not_array = not_array[1:]
    final_array = []
    for element in not_array:
        element = element.strip()
        open_brackets = 1
        result = ""
        index = 0
        while open_brackets > 0:
            if element[index] == '(':
                open_brackets += 1
            elif element[index] == ')':
                open_brackets -= 1
            if open_brackets > 0:
                result += element[index]
            index += 1

        final_array.append(result.strip())
    return final_array


def apply_negation(topic):
    array = negate_topic(topic)
    for el in array:
        original_el = el
        for entity in enitities_exclude_not:
            el = el.replace(entity, negate_mapping[entity])
            
        topic = topic.replace(original_el, el)
    return topic


negated_topics = np.vectorize(apply_negation)(topics)

negated_entities = set([word[1:]
               for t in negated_topics for word in t.split() if word.isupper()])

entities_copy = negated_entities.copy()
entities_copy.remove('PIZZAORDER')
entities_copy.remove('NOT')
entities_copy.remove('COMPLEX_TOPPING')
entities_copy.remove('DRINKORDER')
entities_copy.remove('ORDER')

# np.save('negated_entities.npy', list(entities_copy))


def get_entity(topic):
    words_array=topic.split()
    result=[]
    current_entity='O'
    for word in words_array:
        if word.startswith("("):
            if word[1:] in entities_copy:
                current_entity=word[1:]
            else:
                current_entity='O'
        elif word.startswith(")"):
            current_entity='O'
        else:
            result.append(current_entity)
    return result
        

entities_to_words_not_processed=[
    get_entity(topic) for topic in tqdm(negated_topics, desc='Extracting entities')
]

entities_to_words_not_processed=np.array(entities_to_words_not_processed, dtype=object)

np.save('dev_entities_to_words_not_processed.npy',entities_to_words_not_processed)