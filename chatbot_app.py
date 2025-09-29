
# CUSTOMER SUPPORT CHATBOT WITH DOCUMENT TRAINING
# ---------------------------------------------------
# This bot answers user queries based on an uploaded PDF document.

import streamlit as st
import logging
import re
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import torch
import random
from transformers import pipeline

# -----------------------------
# Logging setup
# -----------------------------
# Logs bot activities to a file for transparency
logging.basicConfig(
    filename="support_bot_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------
# Helper Function: To Load PDF
# -----------------------------


def load_pdf(uploaded_file):
    """
    Reads a PDF file and extracts all text.

    Arguments: uploaded_file: PDF file uploaded by the user via Streamlit.

    Returns: text: Full text content of the PDF.
    """
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logging.info("PDF loaded successfully.")
        return text
    except Exception as e:
        logging.error(f"PDF load error: {e}")
        st.error("Failed to read PDF.")
        return ""

# -----------------------------
# Helper Function: Chunk Document by Headings
# -----------------------------


def chunk_by_headings(text: str):
    """
    Splits the document into sections based on headings. Each section contains a heading + content.

    Arguments: text: The full document text.

    Returns: List of sections (chunks) for embedding and retrieval.
    """
    # Regex pattern assumes headings start with capital letters and are on their own line
    pattern = r"(?m)^(?P<header>[A-Z][A-Za-z ]{2,})\n"
    matches = list(re.finditer(pattern, text))

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        chunks.append(chunk_text)

    return chunks if chunks else [text]  # fallback if no headings detected

# -----------------------------
# Support Bot Agent Class
# -----------------------------


class SupportBotAgent:
    """
    Main class for managing document QA, semantic search,
    simulated feedback, and response formatting.
    """

    def __init__(self, text: str):

        # Sentence transformer for semantic embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.sections = chunk_by_headings(text)
        self.section_embeddings = self.embedder.encode(
            self.sections, convert_to_tensor=True
        )
        # Hugging Face QA pipeline for extracting precise answers
        self.qa_pipeline = pipeline(
            "question-answering", model="distilbert-base-uncased")
        logging.info("Bot initialized with document and chunked by headings.")

    # -----------------------------
    # Find Most Relevant Section
    # -----------------------------
    def find_relevant_section(self, query: str):
        """
        Computes cosine similarity between query and each document section.

        Arguments: query: User query string.

        Returns:
            best_section: Most relevant section.
            best_score: Similarity score (0-1).
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(
            query_embedding, self.section_embeddings)[0]
        best_idx = int(torch.argmax(similarities))
        best_score = float(similarities[best_idx])
        return self.sections[best_idx], best_score

    # -----------------------------
    # Extract Best Answer Using QA Model
    # -----------------------------
    def extract_best_answer(self, context: str, query: str):
        """
        Uses Hugging Face QA model to extract precise answer from context.

        Arguments:
            context: Section of text where answer may exist.
            query: User query string.

        Returns: answer: Extracted answer from context.
        """
        try:
            result = self.qa_pipeline(question=query, context=context)
            return result['answer']
        except Exception as e:
            logging.error(f"QA extraction error: {e}")
            # Fallback: return first sentence
            sentences = re.split(r'(?<=[.!?])\s+', context)
            return sentences[0] if sentences else context

    # -----------------------------
    # Format Response
    # -----------------------------
    def format_response(self, context: str, answer: str, score: float):
        """
        Formats response with heading + answer + confidence score.

        Arguments:
            context: Full section text.
            answer: Extracted or selected answer.
            score: Similarity/confidence score.

        Returns: formatted_response: Final string for display.
        """
        lines = context.strip().split("\n")
        heading = lines[0].strip() if len(lines) > 1 else ""
        content = answer
        if heading and heading in content:
            content = content.replace(heading, "").strip()
        if heading:
            return f"{heading}: {content} (Retrieved, Confidence: {score:.2f})"
        else:
            return f"{content} (Retrieved, Confidence: {score:.2f})"

    # -----------------------------
    # Main Query Answering Logic
    # -----------------------------
    def answer_query(self, query: str, max_iterations=2):
        """
        Answers user query with semantic search + QA + feedback loop.

        Args:
            query: User question.
            max_iterations: Max number of feedback iterations.

        Returns:
            response: Formatted answer string.
            context: Section used to generate answer.
            score: Similarity score.
        """
        iteration = 0
        while iteration < max_iterations:
            context, score = self.find_relevant_section(query)

            # Low confidence → no relevant info
            if score < 0.3:
                logging.info(
                    f"Query: {query} | Confidence too low ({score:.2f})")
                return "I couldn’t find relevant information in the document. ❌", None, score

            # Extract precise answer using QA model
            answer = self.extract_best_answer(context, query)
            response = self.format_response(context, answer, score)

            # Simulate feedback: 60% good, 20% too vague, 20% not helpful
            feedback = random.choices(
                ["good", "too vague", "not helpful"], weights=[0.6, 0.2, 0.2], k=1
            )[0]
            logging.info(
                f"Query: {query} | Iteration: {iteration+1} | Feedback: {feedback}")

            # Adjust response based on feedback
            if feedback == "too vague":
                response += f"\n\n[Additional info]: {context[:200]}..."
            elif feedback == "not helpful":
                response = context + f"\n(Rephrased, Confidence: {score:.2f})"
            else:
                return response, context, score

            iteration += 1

        # Return final response after max_iterations
        return response, context, score


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Support Bot", page_icon="🤖", layout="centered")
st.title("📄 Serri: Document Support")

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF and initialize bot
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file and not st.session_state.bot:
    with st.spinner("Processing document..."):
        doc_text = load_pdf(uploaded_file)
        if doc_text:
            st.session_state.bot = SupportBotAgent(doc_text)
            st.success("Document loaded successfully!")

# Clear chat button
if st.session_state.bot and st.button("🧹 Clear Chat"):
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and response handling
user_query = st.chat_input("💬 Type your question about the document...")
if user_query:
    # Log user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Check if bot is initialized
    if st.session_state.bot:
        with st.chat_message("assistant"):
            try:
                answer, context, score = st.session_state.bot.answer_query(
                    user_query)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})
            except Exception as e:
                st.error("⚠️ Something went wrong while processing your query.")
                logging.error(f"Error during query handling: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("📎 Please upload a PDF before asking questions.")
