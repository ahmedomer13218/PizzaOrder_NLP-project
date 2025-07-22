import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention, Input, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import json

VOCAB_SIZE = 30522
EMBEDDING_DIM = 128  
MAX_LEN = 30
NUM_CLASSES = 5

def intialize_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, NUM_CLASSES):
    input_seq = tf.keras.layers.Input(shape=(MAX_LEN,), dtype='int32', name="Input_Sequence")

    # Add dropout to embedding layer
    embedding = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, name="Embedding_Layer")(input_seq)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)

    # Add recurrent dropout to LSTM
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, 
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             name="Encoder_LSTM"),
        name="Bidirectional_LSTM"
    )(embedding)

    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

    # Add dropout and regularization to decoder LSTM
    decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       name="Decoder_LSTM")
    decoder_outputs = decoder_lstm(encoder_outputs, initial_state=[state_h, state_c])

    attention = tf.keras.layers.Attention(name="Attention_Layer")([decoder_outputs, encoder_outputs])
    
    # Add dropout after attention
    attention = tf.keras.layers.Dropout(0.2)(attention)

    combined = tf.keras.layers.Concatenate()([decoder_outputs, attention])

    # Add batch normalization and dropout before final layers
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.2)(combined)

    # Pizza Output
    pizza_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(NUM_CLASSES, 
                              activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="Order_Output_Layer"
    )(combined)

    # Define the model with two outputs
    model = tf.keras.Model(inputs=input_seq, outputs=pizza_output, name="Encoder_Decoder_NER")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    return model

model= intialize_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, NUM_CLASSES)

input_ids_train_second=np.load('data/processed/train/input_ids_train_second.npy', allow_pickle=True)
padded_labels_train_second=np.load('data/processed/train/padded_labels_train_second.npy', allow_pickle=True)
input_ids_dev_second=np.load('data/processed/dev/input_ids_dev_second.npy', allow_pickle=True)
padded_labels_dev_second=np.load('data/processed/dev/padded_labels_dev_second.npy', allow_pickle=True)

X_train = input_ids_train_second

X_dev = input_ids_dev_second
y_dev = padded_labels_dev_second

def create_tf_dataset(input_ids, labels):
    dataset = tf.data.Dataset.from_tensor_slices(({"Input_Sequence": input_ids}, labels))
    return dataset.batch(64)


X_train_sampled, _, labels_sampled, _ = train_test_split(
    X_train,
    padded_labels_train_second,
    test_size=0.6,
    random_state=42,
)

# Prepare the new labels
y_train_sampled = labels_sampled

# Create the sampled train dataset
train_dataset_sampled = create_tf_dataset(X_train_sampled, y_train_sampled)
dev_dataset = create_tf_dataset(X_dev, y_dev)



checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/2/model_{epoch:02d}-{val_loss:.2f}.weights.h5',
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
    train_dataset_sampled,
    validation_data=dev_dataset,
    epochs=4,
    callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback, early_stop_callback]
)

model.save('models/shared_encoder_decoder2.keras')

history_dict = history.history
with open('training_history_encoder_decoder2.json', 'w') as f:
    json.dump(history_dict, f)
