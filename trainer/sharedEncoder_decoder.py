import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention, Input, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import json



VOCAB_SIZE = 30522
EMBEDDING_DIM = 128  
MAX_LEN = 30
NUM_CLASSES_PIZZA = 21
NUM_CLASSES_DRINKS = 21

def intialize_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, NUM_CLASSES_PIZZA, NUM_CLASSES_DRINKS):
    input_seq = tf.keras.layers.Input(shape=(MAX_LEN,), dtype='int32', name="Input_Sequence")

    # Add dropout to embedding layer
    embedding = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, name="Embedding_Layer")(input_seq)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    
    # Add Gaussian noise layer
    noisy_embedding = tf.keras.layers.GaussianNoise(0.1)(embedding)

    # Add recurrent dropout to LSTM
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, return_state=True, 
             dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=tf.keras.regularizers.l2(0.01),
             name="Encoder_LSTM"),
        name="Bidirectional_LSTM"
    )(noisy_embedding)  # Using noisy embedding instead of regular embedding

    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

    # Rest of the model remains the same
    decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True,
                       dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       name="Decoder_LSTM")
    decoder_outputs = decoder_lstm(encoder_outputs, initial_state=[state_h, state_c])

    attention = tf.keras.layers.Attention(name="Attention_Layer")([decoder_outputs, encoder_outputs])
    attention = tf.keras.layers.Dropout(0.2)(attention)

    combined = tf.keras.layers.Concatenate()([decoder_outputs, attention])
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.2)(combined)

    pizza_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(NUM_CLASSES_PIZZA, 
                              activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="Pizza_Output_Layer"
    )(combined)

    drinks_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(NUM_CLASSES_DRINKS, 
                              activation="softmax", 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        name="Drinks_Output_Layer"
    )(combined)

    model = tf.keras.Model(inputs=input_seq, outputs=[pizza_output, drinks_output], name="Hybrid_Encoder_Decoder_NER")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss={
                      "Pizza_Output_Layer": "sparse_categorical_crossentropy",
                      "Drinks_Output_Layer": "sparse_categorical_crossentropy"
                  },
                  metrics={
                      "Pizza_Output_Layer": "accuracy",
                      "Drinks_Output_Layer": "accuracy"
                  })

    model.summary()

    return model

model= intialize_model(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, NUM_CLASSES_PIZZA,NUM_CLASSES_DRINKS)



input_ids_train=np.load('data/processed/train/input_ids_pizza_train.npy', allow_pickle=True)
padded_labels_drink_train=np.load('data/processed/train/padded_labels_drink_train.npy', allow_pickle=True)
padded_labels_pizza_train=np.load('data/processed/train/padded_labels_pizza_train.npy', allow_pickle=True)
input_ids_dev=np.load('data/processed/dev/input_ids_drink_dev.npy', allow_pickle=True)
padded_labels_drink_dev=np.load('data/processed/dev/padded_labels_drink_dev.npy', allow_pickle=True)
padded_labels_pizza_dev=np.load('data/processed/dev/padded_labels_pizza_dev.npy', allow_pickle=True)
input_ids_test=np.load('data/processed/test/input_ids_drink_test.npy', allow_pickle=True)
padded_labels_drink_test=np.load('data/processed/test/padded_labels_drink_test.npy', allow_pickle=True)
padded_labels_pizza_test=np.load('data/processed/test/padded_labels_pizza_test.npy', allow_pickle=True)

X_train = input_ids_train
y_train_pizza = padded_labels_pizza_train
y_train_drinks = padded_labels_drink_train
# y_train = {
#     "Pizza_Output_Layer": padded_labels_pizza_train,
#     "Drinks_Output_Layer": padded_labels_drink_train
# }

X_dev = input_ids_dev
y_dev = {
    "Pizza_Output_Layer": padded_labels_pizza_dev,
    "Drinks_Output_Layer": padded_labels_drink_dev
}

def create_tf_dataset(input_ids, labels):
    dataset = tf.data.Dataset.from_tensor_slices(({"Input_Sequence": input_ids}, labels))
    return dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)


stratify_labels = list(zip(padded_labels_pizza_train[:, 0], padded_labels_drink_train[:, 0]))

# Split the data, keeping 50% of the original data with balanced stratification
X_train_sampled, _, pizza_labels_sampled, _, drink_labels_sampled, _ = train_test_split(
    X_train,  # Input sequences
    padded_labels_pizza_train,  # Pizza labels
    padded_labels_drink_train,  # Drinks labels
    test_size=0.95,  # Retain 50% of the data
    stratify=stratify_labels,  # Use combined labels for balance
    random_state=42  # For reproducibility
)

# Prepare the new labels
# y_train_sampled = {
#     "Pizza_Output_Layer": np.array(np.concatenate([np.array(pizza_labels_sampled),padded_labels_pizza_dev, padded_labels_pizza_test],axis=0)),
#     "Drinks_Output_Layer": np.array(np.concatenate([np.array(drink_labels_sampled),padded_labels_drink_dev, padded_labels_drink_test],axis=0))
# }

y_train_sampled = {
    "Pizza_Output_Layer": np.array(pizza_labels_sampled),
    "Drinks_Output_Layer": np.array(drink_labels_sampled)
}

# Create the sampled train dataset
# train_dataset_sampled = create_tf_dataset(np.concatenate([X_train_sampled,input_ids_dev,input_ids_test],axis=0), y_train_sampled)
train_dataset_sampled = create_tf_dataset(X_train_sampled, y_train_sampled)

dev_dataset = create_tf_dataset(X_dev, y_dev)



checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_noise2_{epoch:02d}-{val_loss:.2f}.weights.h5',
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
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback, early_stop_callback]
)

model.save('models/shared_encoder_decoder02_noise2.keras')

history_dict = history.history
with open('training_history_encoder_decoder2.json', 'w') as f:
    json.dump(history_dict, f)
