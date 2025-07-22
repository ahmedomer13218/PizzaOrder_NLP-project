from typing import Literal, Union
import json
from collections import defaultdict
import tensorflow as tf
from tqdm import tqdm


def load_data(file_path,type: Literal['train', 'dev', 'test']):
    if type == 'train':
        data_train = []
        with open(file_path, 'r') as f:
            for line in f:
                data_train.append(json.loads(line))
        return data_train
    elif type == 'dev':
        with open(file_path, 'r') as file:
            data_dev = json.load(file)
        return data_dev
    elif type == 'test':
        data_test = []
        with open(file_path, 'r') as f:
            for line in f:
                data_test.append(json.loads(line))
        return data_test
    else:
        raise ValueError('Invalid type')

def extract_text(data,type: Literal['train', 'dev', 'test']):
    if type =='train':
        return data['train.SRC'], data['train.EXR'], data['train.TOP'], data['train.TOP-DECOUPLED']
    elif type == 'dev':
        return data['dev.SRC'], data['dev.EXR'], data['dev.TOP'], data['dev.PCFG_ERR']
    elif type == 'test':
        return data['test.SRC'], data['test.EXR'], data['test.TOP'], data['test.PCFG_ERR']
    else:
        raise ValueError('Invalid type')   
    
    

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


def apply_negation(topic,negate_mapping,enitities_exclude_not):
    array = negate_topic(topic)
    for el in array:
        original_el = el
        for entity in enitities_exclude_not:
            el = el.replace(entity, negate_mapping[entity])
            
        topic = topic.replace(original_el, el)
    return topic

def get_entity(topic,mapping):
    # print(mapping)
    words_array=topic.split()
    result=[]
    current_entity='O'
    first=False
    for word in words_array:
        if word.startswith("("):
            if word[1:] in mapping:
                current_entity=word[1:]
                first=True
            else:
                current_entity='O'
        elif word.startswith(")"):
            current_entity='O'
        else:
            if current_entity=='O':
                result.append(current_entity)
            else:
                if first:
                    result.append('B-'+current_entity)
                    first=False
                else:
                    result.append('I-'+current_entity)    
    return result
        
def get_entity2(topic,mapping,mapping2):
    words_array=topic.split()

    result=[]
    current_entity='O'
    first=False
    open_brackets=0
    for word in words_array:
        if word.startswith("("):
            if word[1:] in mapping2:
                open_brackets+=1
            else:
                if word[1:] in mapping:
                    current_entity=word[1:]
                    first=True
                elif 'NOT' not in word[1:] and 'COMPLEX_TOPPING' not in word[1:]:
                    current_entity='O'
        elif word.startswith(")"):
            if open_brackets>0:
                open_brackets-=1
            else:
                current_entity='O'
        else:
            if current_entity=='O':
                result.append(current_entity)
            else:
                if first:
                    result.append('B-'+current_entity)
                    first=False
                else:
                    result.append('I-'+current_entity)    
    return result

def extract_unique_words(sentences, labels):
    entity_words = defaultdict(set)
    for sentence, label_seq in zip(sentences, labels):
        for word, label in zip(sentence.split(), label_seq):
            if label != "O":
                entity_type = label.split("-")[-1]
                entity_words[entity_type].add(word)

    return entity_words

def map_label_sequences(labels):
    sequence_map = {}
    sequence_indices = []
    current_index = 0

    for label_seq in labels:
        label_tuple = tuple(label_seq)
        
        if label_tuple not in sequence_map:
            sequence_map[label_tuple] = current_index
            current_index += 1
        
        sequence_indices.append(sequence_map[label_tuple])

    return sequence_indices, sequence_map


def prepare_data(sentences, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    with tf.device('/CPU:0'):
        for sentence in tqdm(sentences, desc="Tokenizing sentences"):
            encoding = tokenizer(sentence,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=max_length,
                                 return_tensors="tf")  # Use "tf" to return TensorFlow tensors

            input_ids.append(encoding["input_ids"])  # TensorFlow tensor
            attention_masks.append(encoding["attention_mask"])  # TensorFlow tensor
    
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    labels = tf.convert_to_tensor(labels)

    return input_ids, attention_masks, labels


def padding_labels(labels, max_len):
    for i in range(len(labels)):
        if len(labels[i]) < max_len:
            labels[i] = labels[i] + ['0'] * (max_len - len(labels[i]))
        else:
            labels[i] = labels[i][:max_len]
    return labels