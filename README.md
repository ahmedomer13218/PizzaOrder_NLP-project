# 🍕 NLP-Based Pizza Order semantic parsing
## Dataset
(https://github.com/amazon-science/pizza-semantic-parsing-dataset/tree/main)
## 🧠 Overview

This project focuses on building an **NLP system** capable of extracting and parsing structured entities from natural language food orders, specifically pizza and drink orders. The aim is to convert user instructions into machine-readable formats for more accurate and efficient order processing in online food delivery systems.

---
## 🧩 Quick Pipeline Overview

<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-2.jpg" width="600"/>

---

## 🧹 1. Preprocessing
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-3.jpg" width="600"/>

## 🏷️ 2. Process Labels
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-4.jpg" width="600"/>
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-5.jpg" width="600"/>

## 🧾 3. Tokenization
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-6.jpg" width="600"/>

## 📐 4. Feature Extraction
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-7.jpg" width="600"/>

## 🔍 5. Modeling
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-8.jpg" width="600"/>
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-9.jpg" width="600"/>
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-10.jpg" width="600"/>

## 🔗 6. Combining
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-11.jpg" width="600"/>

## 🚀 7. Demo
<img src="Presentation/NLP_Project_Images/Pizza Orders NLP Project-12.jpg" width="600"/>


---
## 🏗️ Project Pipeline in details

### 1. 🔍 Data Preprocessing

We explored several strategies and finalized a robust pipeline:

**Tried Approaches**
- Spell Checker
- Contractions Expansion
- Strip Extra Spaces
- Lowercasing
- Punctuation Removal
- Lemmatization

**Final Pipeline**
- Spell Checker ✅
- Strip Extra Spaces ✅
- Lowercasing ✅
- Punctuation Removal ✅

---

### 2. 🏷️ Label Preparation

**Tried**
- Regular Expressions (Regex)

**Final**
- Custom Function tailored for parsing domain-specific entities (e.g., toppings, styles, drink types)

---

### 3. 🔡 Tokenization

Used **`BertTokenizer`** to ensure consistency with the transformer-based model architectures.

---

### 4. 🧪 Feature Extraction

We explored and compared several feature extraction techniques:

- Bag of Words (BoW)
- TF-IDF
- Word2Vec
- Contextual embeddings (BERT)
- Trainable Embeddings (for LSTM-based models)

---

### 5. 🧠 Model Architectures

We built multiple models ranging from classical ML to deep learning:

#### 🔁 RNN-Based Architectures

**Model 1** – *Single Entity Type (Pizza or Drink)*
- Trainable embeddings
- BiLSTM (128 units)
- LSTM decoder (256 units)
- Attention Layer

**Model 2** – *Shared Encoder, Two Decoders*
- Trainable embeddings
- BiLSTM Encoder (128 units) + Dropout (0.2)
- Two LSTM Decoders (256 units) + Dropout
- Attention Layer
- Dual Dense Output for Pizza & Drink

**Model 3** – *Specialized for PIZZAORDER and DRINKORDER*
- Same as Model 2 but separated architectures

#### 🔀 Transformer-Based

**Model 4**
- TFBertModel (frozen)
- Two BiLSTM layers
- Dense Output Layer

---

### 6. 🏋️ Model Training

- Preprocessed input using the final pipeline
- Labels padded to max 30 tokens
- Tokenized via `BertTokenizer` (input IDs + attention masks)
- Training with callbacks:
  - `ModelCheckpoint`
  - `ReduceLROnPlateau`
  - `EarlyStopping`
  - `TensorBoard`

---

### 7. 🧾 Evaluation & Testing

- Evaluated using **Exact Match Accuracy**
- Final model ensemble includes:
  - Shared encoder–decoder for both pizza and drink
  - Separate encoder–decoder for either PIZZAORDER or DRINKORDER
- Combined predictions to generate final **TOP format** outputs

---

## 🧪 Dataset

- **Training Samples**: 2,456,446
- **Entities**: Toppings, Sizes, Numbers, Styles, Drink Types, etc.
- **Annotation Format**: Hierarchical structure (e.g., `(ORDER (PIZZAORDER ...))`)

---

## 📈 Results

We focused on:
- Validation Accuracy
- Exact Match Accuracy (EM)
- Overfitting reduction

Final results were submitted via **Kaggle** for ranking based on **Modulo Sibling Order EM Score**.

---

## 🧩 Data Augmentation

To diversify training:
- Extracted all unique entity words
- Combined them to generate synthetic training sequences

---

## 🧪 Testing Demo

A demo is available where users can input natural pizza orders and receive structured JSON-like outputs representing the parsed order tree.


---


