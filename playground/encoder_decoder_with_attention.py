import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Attention, Input, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import json

# VOCAB_SIZE = 30522
# EMBEDDING_DIM = 128  
# MAX_LEN = 30
# NUM_CLASSES = 33

# def intialize_model(VOCAB_SIZE,EMBEDDING_DIM,MAX_LEN,NUM_CLASSES):
#     input_seq = Input(shape=(MAX_LEN,), dtype='int32', name="Input_Sequence")

#     embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, name="Embedding_Layer")(input_seq)

#     encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
#         LSTM(128, return_sequences=True, return_state=True, name="Encoder_LSTM"),
#         name="Bidirectional_LSTM"
#     )(embedding)

#     state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
#     state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

#     decoder_lstm = LSTM(256, return_sequences=True, name="Decoder_LSTM")
#     decoder_outputs = decoder_lstm(encoder_outputs, initial_state=[state_h, state_c])

#     attention = Attention(name="Attention_Layer")([decoder_outputs, encoder_outputs])

#     combined = tf.keras.layers.Concatenate()([decoder_outputs, attention])

#     output = TimeDistributed(Dense(NUM_CLASSES, activation="softmax"), name="Output_Layer")(combined)

#     model = Model(inputs=input_seq, outputs=output, name="Encoder_Decoder_NER")
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#     model.summary()

#     return model

# model = intialize_model(VOCAB_SIZE,EMBEDDING_DIM,MAX_LEN,NUM_CLASSES)

input_ids_train=np.load('input_ids_train_position.npy', allow_pickle=True)
attention_masks_train=np.load('attention_masks_train_position.npy', allow_pickle=True)
padded_labels_train=np.load('padded_labels_train_position.npy', allow_pickle=True)
input_ids_dev=np.load('input_ids_dev_position.npy', allow_pickle=True)
attention_masks_dev=np.load('attention_masks_dev_position.npy', allow_pickle=True)
padded_labels_dev=np.load('padded_labels_dev_position.npy', allow_pickle=True)


# Generate random indices for shuffling
train_size = len(input_ids_train)
indices = np.random.permutation(train_size)

# Calculate split point (10% of train data)
split_point = int(train_size * 0.1)

# Split indices
transfer_indices = indices[:split_point]

# Transfer 10% from train to dev
input_ids_dev = np.concatenate([input_ids_dev, input_ids_train[transfer_indices]])
attention_masks_dev = np.concatenate([attention_masks_dev, attention_masks_train[transfer_indices]])
padded_labels_dev = np.concatenate([padded_labels_dev, padded_labels_train[transfer_indices]])

# Keep remaining 90% for train
keep_indices = indices[split_point:]
input_ids_train = input_ids_train[keep_indices]
attention_masks_train = attention_masks_train[keep_indices]
padded_labels_train = padded_labels_train[keep_indices]

# Shuffle dev set
dev_indices = np.random.permutation(len(input_ids_dev))
input_ids_dev = input_ids_dev[dev_indices]
attention_masks_dev = attention_masks_dev[dev_indices]
padded_labels_dev = padded_labels_dev[dev_indices]

np.save('input_ids_train_position_shuffled.npy', input_ids_train)
np.save('attention_masks_train_position_shuffled.npy', attention_masks_train)
np.save('padded_labels_train_position_shuffled.npy', padded_labels_train)
np.save('input_ids_dev_position_shuffled.npy', input_ids_dev)
np.save('attention_masks_dev_position_shuffled.npy', attention_masks_dev)
np.save('padded_labels_dev_position_shuffled.npy', padded_labels_dev)


# def create_tf_dataset_with_attention(input_ids, attention_mask, labels):
#     dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, labels))
#     return dataset.batch(32)

# def create_tf_dataset(input_ids, labels):
#     dataset = tf.data.Dataset.from_tensor_slices(({"Input_Sequence": input_ids}, labels))
#     return dataset.batch(32)


# train_dataset = create_tf_dataset(input_ids_train, padded_labels_train)
# dev_dataset = create_tf_dataset(input_ids_dev, padded_labels_dev)

# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='model_checkpoint.weights.h5',
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True
# )

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

# reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.2,
#     patience=2,
#     min_lr=0.00001
# )

# early_stop_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=3,
#     restore_best_weights=True
# )


# history = model.fit(
#     train_dataset,
#     validation_data=dev_dataset,
#     epochs=6,
#     callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback, early_stop_callback]
# )

# model.save('encoder_decoder_with_positions.keras')

# history_dict = history.history
# with open('training_history_encoder_decoder.json', 'w') as f:
#     json.dump(history_dict, f)

