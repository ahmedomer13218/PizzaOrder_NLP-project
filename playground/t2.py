import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the Encoder-Decoder Model for NER with LSTM

class EncoderDecoderNERModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(EncoderDecoderNERModel, self).__init__()
        
        # Embedding Layer for token inputs (use pre-trained embeddings if available)
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        
        # LSTM Encoder
        self.encoder_lstm = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True))
        
        # Decoder LSTM (Could also be used for other purposes, here just for demonstration)
        self.decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True)
        
        # Dense Layer for Classification (NER task)
        self.dense = layers.Dense(num_labels, activation="softmax")

    def call(self, inputs):
        # Embedding layer
        embedded = self.embedding(inputs)
        
        # Pass through the LSTM encoder
        encoder_output = self.encoder_lstm(embedded)
        
        # Pass through the decoder LSTM
        decoder_output = self.decoder_lstm(encoder_output)
        
        # Final Dense layer to predict labels for each token
        output = self.dense(decoder_output)
        
        return output

# Set parameters
vocab_size = 30522  # For BERT's tokenizer
embedding_dim = 128
hidden_dim = 64
num_labels = 11

# Initialize the model
model = EncoderDecoderNERModel(vocab_size, embedding_dim, hidden_dim, num_labels)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

def preprocess_data(sentences, labels, tokenizer, max_length, num_labels,entities_id):
    tokenized_inputs = []
    tokenized_labels = []
    
    for sentence, label in zip(sentences[:10000], labels[:10000]):
        # print(sentence)
        encoding = tokenizer(sentence, truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True)
        # print(encoding)
        tokenized_input = encoding['input_ids']
        # print(tokenized_input)
        
        aligned_labels = ['0']
        for i, word in enumerate(sentence):
            subwords = tokenizer.tokenize(word)
            for subword in subwords:
                aligned_labels.append(entities_id[label[i]])  # Assign the same label to subword tokens
        
        # Pad the labels
        while len(aligned_labels) < max_length:
            aligned_labels.append(0)  # Padding label (can be O for NER tasks)
        
        tokenized_inputs.append(tokenized_input)
        tokenized_labels.append(aligned_labels)
    
    # Convert to numpy arrays
    tokenized_inputs = np.array(tokenized_inputs)
    tokenized_labels = np.array(tokenized_labels, dtype=np.int64)
    
    return tokenized_inputs, tokenized_labels

sentences=np.load('text_2d.npy', allow_pickle=True)
ss=[" ".join(s) for s in sentences]
labels=np.load('entities_to_words_not_processed.npy', allow_pickle=True)

max_len=30

entities=np.load('negated_entities.npy', allow_pickle=True)
entities_id = {e.item(): i+1 for i, e in enumerate(entities)}
entities_id['0']=0
entities_id['O']=0

reversed_entities_id = {v: k for k, v in entities_id.items() if k != 0}
reversed_entities_id[0]='O'


# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids, encoded_labels = preprocess_data(sentences, labels, tokenizer, max_len, num_labels,entities_id)
print(input_ids.dtype)
print(encoded_labels.dtype) 

# print(input_ids, encoded_labels)
# print(encoded_labels.shape)
# print(input_ids.shape)
# for i in input_ids:
#     print(tokenizer.decode(i))
# print(encoded_labels)

# Tokenize the words (use BERT's tokenizer)
# input_ids = [tokenizer(s, padding=True, truncation=True, return_tensors="np")['input_ids'] for s in ss[:5]]
# print(ss[:5])
# print(input_ids)
# for i in input_ids:
#     print(tokenizer.decode(i))

# # Convert labels to integers (assuming you're using an integer encoding for labels)
# label_map = {"O": 0, "B-PER": 1, "I-PER": 2}
# encoded_labels = np.array([label_map[label] for label in labels])

# # Pad sequences to the same length
# input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post', value=0)
# encoded_labels = tf.keras.preprocessing.sequence.pad_sequences([encoded_labels], padding='post', value=0)

# # # Ensure input shape is correct (2D tensor: batch_size, sequence_length)
# input_ids = np.array(input_ids)
# encoded_labels = np.array(encoded_labels)


# # # Reshape labels if needed for sparse categorical crossentropy
# encoded_labels = np.expand_dims(encoded_labels, -1)
# print(input_ids)
# print(encoded_labels)

# # Train the model
model.fit(input_ids, encoded_labels, epochs=5, batch_size=32)


# Example sentence
sentence = ["I", "would", "like", "a", "large", "pizza"]

# Tokenize and pad
encoded_input = tokenizer(sentence, 
                          truncation=True, 
                          padding="max_length", 
                          max_length=30, 
                          is_split_into_words=True)

input_ids = np.array([encoded_input["input_ids"]]) 

raw_predictions = model.predict(input_ids)

predicted_label_indices = tf.argmax(raw_predictions, axis=-1).numpy()  # Shape: (batch_size, seq_len)

predicted_label_indices = predicted_label_indices[0]
output=predicted_label_indices[1:len(sentence)+1]
output_entities=[reversed_entities_id[i] for i in output]

for word, label in zip(sentence, output_entities):
    print(f"{word}: {label}")
