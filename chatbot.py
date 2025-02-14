import streamlit as st
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch  # For web search
from PIL import Image
import requests
from io import BytesIO
import time  # For streaming effect

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBcWys8yTQCTpesAgvxle7PDEwg6si-H0c"
genai.configure(api_key=GEMINI_API_KEY)

# Configure SerpAPI
SERPAPI_KEY = "983858ca65811d41b677aa8e6f0e6a2e24b3fa34ecc4c9e2dff82bf139ab016d"

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
art_projects = [
    "How to make a paper mache mask: Mix glue and water, tear newspaper into strips, dip in the mixture, and layer on a balloon. Let it dry, then paint.",
    "How to create a cardboard castle: Cut cardboard into castle shapes, glue them together, and paint with colors to decorate.",
    "How to make a leaf painting: Collect leaves, apply paint to them, and press onto paper to create prints.",
    "How to craft a clay pot: Shape clay into a small pot, let it dry, and paint it with vibrant colors.",
    "How to design a friendship bracelet: Use colorful threads, braid or knot them into patterns, and tie the ends securely.",
    "How to create a mosaic art piece: Break old tiles into small pieces, arrange them on a board, and glue them into a design.",
    "How to make a nature collage: Collect dried flowers, leaves, and twigs, then glue them onto a canvas to create a natural scene.",
    "How to do a salt painting: Draw with glue on paper, sprinkle salt, let it dry, then add watercolor for a spreading effect.",
    "How to create a recycled bottle vase: Take an old plastic bottle, cut it into a vase shape, and paint or decorate it.",
    "How to make an origami animal: Take a square sheet of paper, follow folding steps, and create an animal shape."
]

# Convert documents to embeddings
embeddings = embed_model.encode(art_projects, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean distance)
index.add(embeddings)

# Streamlit UI
st.set_page_config(page_title="Chatbot", layout="wide")

st.title("📌 Art Project Chatbot")
st.markdown("Ask me about art projects!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to retrieve relevant documents
def retrieve_relevant_docs(query, top_k=2, threshold=0.5):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    relevant_docs = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:  # Lower distance means more relevant
            relevant_docs.append(art_projects[i])

    # If no relevant docs found, fallback to web search
    if not relevant_docs:
        return search_web(query)

    return relevant_docs

# Function to search the web for images
def search_web(query):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "tbm": "isch"  # Search for images
    })
    results = search.get_dict()

    # Extract first image URL
    if "images_results" in results:
        first_image_url = results["images_results"][0]["original"]
        return first_image_url
    return None

# Function to generate response using Gemini API with chat history
def generate_response(query, retrieved_docs):
    context = "\n".join(retrieved_docs) if isinstance(retrieved_docs, list) else ""
    source = "Dataset" if context else "Web Search"

    # Include last 3 messages from chat history for context
    chat_history = "\n".join([f"User: {chat['query']}\nBot: {chat['response']}" for chat in st.session_state.chat_history[-3:]])

    prompt = f"""
    You are an AI assistant specializing in school-level syllabus. Answer the following query with step-by-step instructions.
    - Instead of giving direct answers, provide the best approach to achieve the result.
    - Maintain a formal and interactive tone for general queries (greetings, farewells).
    - Refer to the previous conversation for context.

    Previous Conversation:
    {chat_history}

    Context ({source}):
    {context}

    Question: {query}
    Answer:
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return response.text if response.text else "I couldn't find the exact steps, but you can try searching online."

# Function to display response word by word
def stream_response(response_text):
    response_placeholder = st.empty()
    full_response = ""

    for word in response_text.split():
        full_response += word + " "
        response_placeholder.write(full_response)  # Stream each word
        time.sleep(0.05)  # Delay for effect

# User Input
query = st.chat_input("Ask a question about art projects")
if query:
    retrieved_docs = retrieve_relevant_docs(query)
    response = generate_response(query, retrieved_docs)

    # Fetch image if available
    image_url = search_web(query)

    # Store in session state
    st.session_state.chat_history.append({"query": query, "response": response, "image": image_url})

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['query']}")

    with st.chat_message("assistant"):
        st.markdown(f"**Bot:**")
        stream_response(chat["response"])  # Stream response word by word

        # Display image if available
        if chat["image"]:
            try:
                img = Image.open(BytesIO(requests.get(chat["image"]).content))
                st.image(img, caption="Relevant Image", width=300)  # Reduced width
            except Exception as e:
                st.error("Could not load image.")
