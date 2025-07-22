import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import json

from sklearn.model_selection import train_test_split

def create_encoder_decoder_model(bert_model, hidden_dim, num_labels_pizza, num_labels_drinks, max_length):
    # Define BERT input layers
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask_pizza = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask_pizza")
    attention_mask_drinks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask_drinks")

    for layer in bert_model.layers:
        layer.trainable = False
    # BERT output
    bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask_pizza)
    bert_embeddings = bert_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

    # Add Gaussian noise layer after BERT embeddings
    x = tf.keras.layers.GaussianNoise(0.1)(bert_embeddings)

    # Add Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(x)
    x = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)(x)

    # Add dropout and regularization to decoder LSTM
    decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       name="Decoder_LSTM")
    decoder_outputs = decoder_lstm(x)

    # Project both tensors to the same dimension (256)
    x_projected = tf.keras.layers.Dense(256)(x)
    attention_pizza = tf.keras.layers.Attention(name="Attention_Layer_Pizza")([decoder_outputs, x_projected])
    attention_drinks = tf.keras.layers.Attention(name="Attention_Layer_Drinks")([decoder_outputs, x_projected])
    
    # Add dropout after attention
    attention_pizza = tf.keras.layers.Dropout(0.2)(attention_pizza)
    attention_drinks = tf.keras.layers.Dropout(0.2)(attention_drinks)

    combined_pizza = tf.keras.layers.Concatenate()([decoder_outputs, attention_pizza])
    combined_drinks = tf.keras.layers.Concatenate()([decoder_outputs, attention_drinks])

    # Add batch normalization and dropout before final layers
    combined_pizza = tf.keras.layers.BatchNormalization()(combined_pizza)
    combined_pizza = tf.keras.layers.Dropout(0.2)(combined_pizza)
    combined_drinks = tf.keras.layers.BatchNormalization()(combined_drinks)
    combined_drinks = tf.keras.layers.Dropout(0.2)(combined_drinks)

    # Pizza Output
    pizza_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_labels_pizza, 
                              activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="Pizza_Output_Layer"
    )(combined_pizza)

    # Drinks Output
    drinks_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_labels_drinks, 
                              activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="Drinks_Output_Layer"
    )(combined_drinks)

    # Define the model with two outputs
    model = tf.keras.Model(inputs=[input_ids, attention_mask_pizza, attention_mask_drinks], outputs=[pizza_output, drinks_output], name="Hybrid_Encoder_Decoder_NER")

    return model

    
    
vocab_size = 30522  # this is the size of the BERT tokenizer
embedding_dim = 128
hidden_dim = 64
num_labels = 33
max_len=30 # max length of the input sequences is 25


# entities=np.load('full_negate_entities.npy', allow_pickle=True)
# entities_id = {e.item(): i+1 for i, e in enumerate(entities)}
# entities_id['0']=0
# entities_id['O']=0

# reversed_entities_id = {v: k for k, v in entities_id.items() if k != 0}
# reversed_entities_id[0]='O'

# print(entities_id)
# print(reversed_entities_id)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

model = create_encoder_decoder_model(bert_model, hidden_dim, num_labels_pizza=21,num_labels_drinks=21, max_length=max_len)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


checkpoint_path = "./model_checkpoint.h5"
model.load_weights(checkpoint_path)

model.summary()

from tensorflow.keras.models import load_model


model2 = load_model('./models/shared_encoder_decoder2.keras')

model2.summary()


input_ids_pizza_test = np.load('data/processed/test/input_ids_pizza_test.npy')

padded_labels_pizza_test = np.load('data/processed/test/padded_labels_pizza_test.npy')
padded_labels_drink_test = np.load('data/processed/test/padded_labels_drink_test.npy')

attention_masks_drink_test=np.load('data/processed/test/attention_masks_drink_test.npy', allow_pickle=True)
attention_masks_pizza_test=np.load('data/processed/test/attention_masks_pizza_test.npy', allow_pickle=True)

padded_labels_test_second = np.load('data/processed/test/padded_labels_test_second.npy')


processed_test_sentences = np.load(
    'data/processed/test/processed_test_sentences.npy')

X_test = {"input_ids": input_ids_pizza_test, "attention_mask_pizza": attention_masks_pizza_test, "attention_mask_drinks": attention_masks_drink_test}


predictions1 = np.array(model.predict(X_test))

predictions2 = np.array(model2.predict(input_ids_pizza_test))


entity_labels_drink=np.load('data/processed/entity_labels_drink.npy')
entity_labels_pizza=np.load('data/processed/entity_labels_pizza.npy')
enitity_labels_second=np.load('data/processed/entity_labels_second.npy')

print(entity_labels_drink)
print(entity_labels_pizza)
print(enitity_labels_second)

entities_id_drink = {i+1: str(e) for i, e in enumerate(entity_labels_drink)}
entities_id_drink[0] = 'O'
entities_id_drink

entities_id_pizza = {i+1: str(e) for i, e in enumerate(entity_labels_pizza)}
entities_id_pizza[0] = 'O'
entities_id_pizza

enitities_id_second = {i+1: str(e) for i, e in enumerate(enitity_labels_second)}
enitities_id_second[0] = 'O'
enitities_id_second

def get_prediction_entities1(processed_sentences, predictions, entities_id):
    pred_entities = []
    for i in range(predictions.shape[0]):
        sen=[entities_id[np.argmax(predictions[i][j])] for j in range(predictions.shape[1])]
        pred_entities.append(sen[1:len(processed_sentences[i].split())+1])
    return pred_entities


entities_pizza = np.array(get_prediction_entities1(
    processed_test_sentences, predictions1[0], entities_id_pizza), dtype=object)
entities_drink = np.array(get_prediction_entities1(
    processed_test_sentences, predictions1[1], entities_id_drink), dtype=object)
entities_second = np.array(get_prediction_entities1(processed_test_sentences,predictions2, enitities_id_second),dtype=object)

def process_second_entities(entities_second):
    current_entity = 'O'
    processed_entities = []
    for entity in entities_second:
        if entity =='O' and current_entity == 'O':
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


process_second_entities(entities_second[120])


def get_TOP(words, predictions1, predictions2, predictions3):
    words=words.split()
    
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
                    if(predictions1[i].startswith('B-NOT_')):
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


get_TOP(processed_test_sentences[0], entities_pizza[0],
        entities_drink[0], entities_second[0])


processed_entities_second = [process_second_entities(es) for es in entities_second]


tops = [get_TOP(processed_test_sentences[i], entities_pizza[i], entities_drink[i], processed_entities_second[i]) for i in range(len(processed_test_sentences))]


np.save('data/output/tops_ctx.npy', tops)


