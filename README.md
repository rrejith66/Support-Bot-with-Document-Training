# 📄 Support Bot with Document Training

A **Python-based customer support chatbot** that reads a PDF document, answers user queries based on the document content, and refines responses using a simulated feedback loop. The bot demonstrates agentic behavior with iterative improvement and logging.

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
- [Development Decisions](#development-decisions)  
- [Known Issues & Future Improvements](#known-issues--future-improvements)  
- [Project Structure](#project-structure)  
- [License](#license)

---

## Features

- **Document Processing**: Reads and extracts text from PDFs using `PyPDF2`.  
- **Semantic Search**: Uses `sentence-transformers` to find the most relevant sections for a query.  
- **QA Model**: Integrates Hugging Face DistilBERT (`distilbert-base-uncased`) for better understanding of the content.  
- **Feedback Simulation**: Adjusts responses based on simulated feedback (`good`, `too vague`, `not helpful`).  
- **Logging**: Tracks document loading, query handling, similarity scores, and feedback in a log file.  
- **Streamlit UI**: Interactive chat interface with chat history and support for multiple queries.  

---

## Requirements

- Python 3.8 or higher
- Libraries:
  - `streamlit`
  - `sentence-transformers`
  - `transformers`
  - `PyPDF2`
  - `torch`
  - Built-in: `logging`, `re`, `random`

---

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/support-bot.git
cd support-bot
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. **Install dependencies**
```bash
pip install streamlit sentence-transformers transformers PyPDF2 torch
```
4. **Run the Streamlit app**
```bash
streamlit run chatbot_app.py
```

## Usage
- **Upload a PDF document containing FAQs, policies, or instructions.**
- **Type your query in the chat input. Example queries used:**
  - "How do I reset my password?"
  - "What’s the refund policy?"
  - "How can I contact support?"
- **View the response:**
  - Short/factoid queries return the most relevant sentence.
  - Instructional/long queries return the full section.
  - Queries not found in the document return: "I couldn’t find relevant information in the document. ❌"
- **Clear chat history by clicking 🧹 Clear Chat.**
- **All actions and feedback are logged in _support_bot_log.txt_.**

## Development Decisions

### Document Chunking
- Split the PDF text into sections using headings (capitalized words followed by line breaks).  
- Helps semantic search locate relevant sections efficiently.

### Embedding & Similarity Matching
- `sentence-transformers` used for semantic embeddings.  
- Cosine similarity used to find the best-matching section for each query.

### Hugging Face QA Model
- `distilbert-base-uncased` used to extract precise answers from the relevant section.

### Simulated Feedback Loop
- Iteratively refines responses using random feedback.  
  - `"too vague"` → append additional context  
  - `"not helpful"` → rephrase the answer

### Logging
- Tracks all steps: document load, query processing, confidence scores, and feedback.

### Streamlit UI
- Chat-based interface with session state to preserve chat history.

---

## Known Issues & Future Improvements

### Current Limitations
- Only PDF uploads supported (TXT support can be added).  
- Responses may truncate long sections if feedback appends additional info.  
- Feedback is simulated and not based on real user ratings.

### Future Improvements
- Collect real-time user feedback.  
- Support TXT documents and multilingual content.  
- Integrate advanced NLP models (e.g., GPT-3.5/GPT-4) for richer answers.  
- Improve chunking for nested headings, bullet points, and tables.  
- Rank multiple relevant sections for ambiguous queries.

## Project Structure

Support-Bot-with-Document-Training/
│
├── app.py # Main Streamlit application
├── chatbot_app.py # SupportBotAgent class and related logic
├── requirements.txt # Python dependencies
├── support_bot_log.txt # Log file capturing actions and queries
├── README.md # Project documentation
└── sample_docs/ # Example documents (PDF)

## License

- **This project is open-source for demonstration purposes.**

## Author
**Rejith R**
[LinkedIn](https://www.linkedin.com/in/rrejith) 
