from transformers import TFBertForTokenClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
import tensorflow as tf


tokens_entities = np.load('tokens_to_entities.npy', allow_pickle=True)
tokens = [[i[0] for i in t] for t in tokens_entities]
entities = [[i[1] for i in t] for t in tokens_entities]
entity_id = {e: i for i, e in enumerate(
    {'VOLUME', 'DRINKTYPE', 'NOT_STYLE', 'NOT_TOPPING', 'TOPPING', 'STYLE', 'NUMBER', 'QUANTITY', 'SIZE', 'CONTAINERTYPE', 'O'})}
entities_processed = [[entity_id[i] for i in e] for e in entities]

model = TFBertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=11)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(
    [" ".join(t) for t in tokens],
    padding=True,
    truncation=True,
    is_split_into_words=True,  # Important to indicate the tokens are already split
    return_tensors="tf"
)

inputs['labels'] = tf.ragged.constant(entities_processed)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(inputs),  # Pass inputs in dict format to align with BERT's input signature
    inputs['labels']  # Use labels as the target
))

training_args = TrainingArguments(
    output_dir='./results',          # Where to save the model outputs
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size per device
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,                # Log every 10 steps
    evaluation_strategy="epoch",     # Evaluate the model after each epoch
    save_strategy="epoch",           # Save model after each epoch
    load_best_model_at_end=True,     # Load the best model after training
    report_to="tensorboard"          # Report to TensorBoard (optional)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

print('Training...')

trainer.train()
