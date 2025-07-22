import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from transformers import TFBertModel

def custom_objects():
    return {'TFBertModel': TFBertModel}

model = tf.keras.models.load_model('encoder_decoder_ner_model_with_c_emb.keras', custom_objects=custom_objects())
model.summary()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

entities=np.load('negated_entities.npy', allow_pickle=True)
entities_id = {e.item(): i+1 for i, e in enumerate(entities)}
entities_id['0']=0
entities_id['O']=0

reversed_entities_id = {v: k for k, v in entities_id.items() if k != 0}
reversed_entities_id[0]='O'

def get_prediction(sentence):
    sentence = sentence.split()
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
    return output_entities



while True:
    sentence = input("Enter a sentence: ")
    if sentence == 'exit':
        break

    preds=get_prediction(sentence)
    for word, label in zip(sentence.split(), preds):
        print(f"{word}--> {label}")


