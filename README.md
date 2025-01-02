# Text Summarization System

A web-based application for summarizing dialogue or text using a fine-tuned Hugging Face T5 transformer model. This project leverages natural language processing (NLP) techniques to condense long pieces of text into concise summaries, making it suitable for various applications like meeting notes, content curation, and educational tools.

---

## Features

- **Fine-tuned Model**: Uses a fine-tuned T5 transformer model from Hugging Face for high-quality text summarization.
- **Text Preprocessing**: Includes a text cleaning function to preprocess input text by removing unnecessary characters, spaces, and tags.
- **Customizable Summarization**: Generates summaries dynamically using beam search to ensure optimal results.
- **Interactive Interface**: Built with Streamlit for a responsive and user-friendly experience.
- **Device Adaptability**: Automatically runs on GPU if available, otherwise uses CPU.
- **Error Handling**: Validates input and provides meaningful feedback to users.

---

## Technologies Used

- **Python**: Core language for the application.
- **Streamlit**: Framework for building the web interface.
- **Hugging Face Transformers**: Library for the T5 model and tokenizer.
- **Regular Expressions**: For text preprocessing.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-system.git

cd text-summarization-system
text-summarization-system/
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── saved_summary_model/   # Folder containing fine-tuned model files
└── README.md              # Project documentation (this file)
