import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras import layers



def create_encoder_decoder_model(vocab_size, embedding_dim, hidden_dim, num_labels, max_length):
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        
        layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True)),
        
        layers.LSTM(hidden_dim, return_sequences=True),
        
        layers.Dense(num_labels, activation="softmax")
    ])
    
    return model
    
# def preprocess_data(sentences, labels, tokenizer, max_length, num_labels, entities_id):
#     tokenized_inputs = []
#     tokenized_labels = []
    
#     for sentence, label in zip(sentences, labels):
#         encoding = tokenizer(sentence, truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True)
#         tokenized_input = encoding['input_ids']
        
#         aligned_labels = [entities_id['0']]  # [CLS] token
#         for i, word in enumerate(sentence):
#             subwords = tokenizer.tokenize(word)
#             aligned_labels.extend([entities_id[label[i]]] * len(subwords))
        
#         aligned_labels = aligned_labels[:max_length] + [entities_id['0']] * (max_length - len(aligned_labels))
        
#         tokenized_inputs.append(tokenized_input)
#         tokenized_labels.append(aligned_labels)
    
#     return np.array(tokenized_inputs), np.array(tokenized_labels, dtype=np.int64)
    
vocab_size = 30522  # this is the size of the BERT tokenizer
embedding_dim = 128
hidden_dim = 64
num_labels = 17
max_len=30 # max length of the input sequences is 25

# sentences=np.load('text_2d.npy', allow_pickle=True)
# ss=[" ".join(s) for s in sentences]
# labels=np.load('entities_to_words_not_processed.npy', allow_pickle=True)

# def padding(sentences, max_len):
#     for i in range(len(sentences)):
#         if len(sentences[i]) < max_len:
#             sentences[i] = sentences[i] + ['[PAD]'] * (max_len - len(sentences[i]))
#         else:
#             sentences[i] = sentences[i][:max_len]
#     return sentences
# def padding_labels(labels, max_len):
#     for i in range(len(labels)):
#         if len(labels[i]) < max_len:
#             labels[i] = labels[i] + ['0'] * (max_len - len(labels[i]))
#         else:
#             labels[i] = labels[i][:max_len]
#     return labels
# def convert_to_ids(sentence, tokenizer):
#     return tokenizer.convert_tokens_to_ids(sentence)
    

entities=np.load('full_negate_entities.npy', allow_pickle=True)
entities_id = {e.item(): i+1 for i, e in enumerate(entities)}
entities_id['0']=0
entities_id['O']=0

reversed_entities_id = {v: k for k, v in entities_id.items() if k != 0}
reversed_entities_id[0]='O'

# print(entities_id)
# print(reversed_entities_id)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens_to_entities=np.load("dev_tokens_to_entities.npy", allow_pickle=True)

# sentences=[['[CLS]']+[i[0] for i in t]+['[SEP]'] for t in tokens_to_entities ]
# padded_sentences = padding(sentences, max_len)
# input_ids = np.vectorize(lambda sentence: tokenizer.convert_tokens_to_ids(sentence))(padded_sentences)
# np.save('dev_input_ids.npy', input_ids)


# labels=[['0']+[entities_id[i[1]] for i in t]+['0'] for t in tokens_to_entities]
# padded_labels = padding_labels(labels, max_len)
# padded_labels=np.array(padded_labels, dtype=np.int64)
# np.save('dev_padded_labels.npy', padded_labels)

input_ids_train=np.load('input_ids.npy', allow_pickle=True)
padded_labels_train=np.load('padded_labels.npy', allow_pickle=True)
input_ids_dev=np.load('dev_input_ids.npy', allow_pickle=True)
padded_labels_dev=np.load('dev_padded_labels.npy', allow_pickle=True)


# print(input_ids.shape)
# print(padded_labels.shape)



# input_ids, encoded_labels = preprocess_data(sentences[:50000], labels[:50000], tokenizer, max_len, num_labels,entities_id)


model = create_encoder_decoder_model(vocab_size, embedding_dim, hidden_dim, num_labels,30)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # model.fit(input_ids, encoded_labels, epochs=4, batch_size=32)
model.fit(input_ids_train, padded_labels_train,validation_data=(input_ids_dev, padded_labels_dev), epochs=1, batch_size=32)


model.save('encoder_decoder_ner_model_with_dev.keras')