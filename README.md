# 🧠 Text Summarization using PEGASUS (Hugging Face Transformers)

This project demonstrates how to build a **text summarization model** using the **PEGASUS transformer model** fine-tuned on the **SAMSum dataset**. The implementation is done using Hugging Face Transformers, Datasets, and PyTorch.

---

## 🚀 Features

* 📚 Uses **PEGASUS (google/pegasus-cnn_dailymail)** pre-trained model
* 🗂 Dataset: **SAMSum (dialogue summarization)**
* 🔄 Data preprocessing with tokenization
* ⚙️ Fine-tuning using Hugging Face `Trainer`
* 📊 Evaluation-ready pipeline
* 💻 GPU/CPU support

---

## 🛠️ Tech Stack

* Python 🐍
* Hugging Face Transformers 🤗
* Hugging Face Datasets 📦
* PyTorch 🔥
* NLTK
* Matplotlib & Pandas

---

## 📂 Project Structure

```
├── Text_summarizer_Project.ipynb   # Main notebook
├── README.md                      # Project documentation
```

---

## 📊 Dataset

We use the **SAMSum dataset**, which contains conversational dialogues and their summaries.

### Example:

**Dialogue:**

```
A: Hey, are we meeting today?
B: Yes, at 5 PM.
```

**Summary:**

```
They are meeting at 5 PM.
```

---

## ⚙️ Installation

Run the following command to install dependencies:

```bash
pip install transformers datasets evaluate sacrebleu rouge_score nltk accelerate
```

---

## 🧪 How It Works

### 1. Load Dataset

```python
from datasets import load_dataset
dataset = load_dataset("knkarthick/samsum")
```

### 2. Load Model & Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
```

### 3. Preprocessing

* Tokenize dialogues as input
* Tokenize summaries as labels

### 4. Training

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset
)

trainer.train()
```

---

## ⚠️ Important Note

In this project, the **test dataset is used for training** for practice purposes.

For real-world applications, always follow proper dataset splitting:

* `train` → Training
* `validation` → Evaluation
* `test` → Final testing

---

## 📈 Future Improvements

* ✅ Use proper dataset splits
* 📊 Add ROUGE score evaluation
* 🚀 Deploy as an API (FastAPI / Flask)
* 🌐 Build a frontend interface
* ⚡ Optimize training with better hyperparameters

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

---

## 📬 Contact

**Vimalanathan Thanushan**
📧 [thanushaan69@gmail.com](mailto:thanushaan69@gmail.com)
🔗 GitHub: https://github.com/ThanushanVimalanathan

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
