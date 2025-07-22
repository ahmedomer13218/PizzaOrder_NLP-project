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

    # Add Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True))(bert_embeddings)
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


# tokens_to_entities=np.load("dev_tokens_to_entities.npy", allow_pickle=True)

# input_ids_train=np.load('input_ids_train.npy', allow_pickle=True)
# attention_masks_train=np.load('attention_masks_train.npy', allow_pickle=True)
# padded_labels_train=np.load('padded_labels_train.npy', allow_pickle=True)
# input_ids_dev=np.load('input_ids_dev.npy', allow_pickle=True)
# attention_masks_dev=np.load('attention_masks_dev.npy', allow_pickle=True)
# padded_labels_dev=np.load('padded_labels_dev.npy', allow_pickle=True)

# input_ids_train = tf.convert_to_tensor(input_ids_train, dtype=tf.int32)
# attention_masks_train = tf.convert_to_tensor(attention_masks_train, dtype=tf.int32)
# padded_labels_train = tf.convert_to_tensor(padded_labels_train, dtype=tf.int32)

# input_ids_dev = tf.convert_to_tensor(input_ids_dev, dtype=tf.int32)
# attention_masks_dev = tf.convert_to_tensor(attention_masks_dev, dtype=tf.int32)
# padded_labels_dev = tf.convert_to_tensor(padded_labels_dev, dtype=tf.int32)

input_ids_train=np.load('data/processed/train/input_ids_pizza_train.npy', allow_pickle=True)
padded_labels_drink_train=np.load('data/processed/train/padded_labels_drink_train.npy', allow_pickle=True)
padded_labels_pizza_train=np.load('data/processed/train/padded_labels_pizza_train.npy', allow_pickle=True)
attention_masks_drink_train=np.load('data/processed/train/attention_masks_drink_train.npy', allow_pickle=True)
attention_masks_pizza_train=np.load('data/processed/train/attention_masks_pizza_train.npy', allow_pickle=True)
input_ids_dev=np.load('data/processed/dev/input_ids_drink_dev.npy', allow_pickle=True)
padded_labels_drink_dev=np.load('data/processed/dev/padded_labels_drink_dev.npy', allow_pickle=True)
padded_labels_pizza_dev=np.load('data/processed/dev/padded_labels_pizza_dev.npy', allow_pickle=True)
attention_masks_drink_dev=np.load('data/processed/dev/attention_masks_drink_dev.npy', allow_pickle=True)
attention_masks_pizza_dev=np.load('data/processed/dev/attention_masks_pizza_dev.npy', allow_pickle=True)
input_ids_test=np.load('data/processed/test/input_ids_drink_test.npy', allow_pickle=True)
padded_labels_drink_test=np.load('data/processed/test/padded_labels_drink_test.npy', allow_pickle=True)
padded_labels_pizza_test=np.load('data/processed/test/padded_labels_pizza_test.npy', allow_pickle=True)
attention_masks_drink_test=np.load('data/processed/test/attention_masks_drink_test.npy', allow_pickle=True)
attention_masks_pizza_test=np.load('data/processed/test/attention_masks_pizza_test.npy', allow_pickle=True)

X_train = {"input_ids": input_ids_train, "attention_mask_pizza": attention_masks_pizza_train, "attention_mask_drinks": attention_masks_drink_train}
y_train = {
    "Pizza_Output_Layer": padded_labels_pizza_train,
    "Drinks_Output_Layer": padded_labels_drink_train
}

X_dev = {"input_ids": input_ids_dev, "attention_mask_pizza": attention_masks_pizza_dev, "attention_mask_drinks": attention_masks_drink_dev}
y_dev = {
    "Pizza_Output_Layer": padded_labels_pizza_dev,
    "Drinks_Output_Layer": padded_labels_drink_dev
}

def create_tf_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

stratify_labels = list(zip(padded_labels_pizza_train[:, 0], padded_labels_drink_train[:, 0]))

# Split the data with balanced stratification
indices = np.arange(len(input_ids_train))
train_indices, _ = train_test_split(
    indices,
    test_size=0.98,
    stratify=stratify_labels,
    random_state=42
)

X_train_sampled = {
    "input_ids": input_ids_train[train_indices],
    "attention_mask_pizza": attention_masks_pizza_train[train_indices],
    "attention_mask_drinks": attention_masks_drink_train[train_indices]
}
y_pizza_sampled = padded_labels_pizza_train[train_indices]
y_drink_sampled = padded_labels_drink_train[train_indices]

# Combine training data with dev and test data
X_combined = {
    "input_ids": np.concatenate([X_train_sampled["input_ids"], input_ids_dev, input_ids_test]),
    "attention_mask_pizza": np.concatenate([X_train_sampled["attention_mask_pizza"], attention_masks_pizza_dev, attention_masks_pizza_test]),
    "attention_mask_drinks": np.concatenate([X_train_sampled["attention_mask_drinks"], attention_masks_drink_dev, attention_masks_drink_test])
}

y_combined = {
    "Pizza_Output_Layer": np.concatenate([y_pizza_sampled, padded_labels_pizza_dev, padded_labels_pizza_test]),
    "Drinks_Output_Layer": np.concatenate([y_drink_sampled, padded_labels_drink_dev, padded_labels_drink_test])
}

# Create the datasets
train_dataset = create_tf_dataset(X_combined, y_combined)
dev_dataset = create_tf_dataset(X_dev, y_dev)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=dev_dataset,
    epochs=6,
    callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback, early_stop_callback]
)

model.save('ctx.keras')


history_dict = history.history
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

