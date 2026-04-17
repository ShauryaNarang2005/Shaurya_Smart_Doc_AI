from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
import streamlit as st

# 🔐 API Key
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

docs_chunks = []

# 📄 Load PDF
def load_pdf(pdf_path):
    global docs_chunks
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    docs_chunks = [doc.page_content for doc in documents]

# 🔍 Retrieve chunks
def retrieve_relevant_chunks(query):
    query_words = set(query.lower().split())
    scored = []

    for chunk in docs_chunks:
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored[:2]]

    if not top_chunks:
        return "\n\n".join(docs_chunks[:2])

    return "\n\n".join(top_chunks)

# 🌐 Web search (STABLE)
def get_web_context(query):
    try:
        results_text = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            for r in results:
                results_text.append(r["body"])
        return "\n".join(results_text)
    except:
        return ""

# 🧠 Improve query
def enhance_query(query):
    if len(query.split()) < 4:
        return query + " explain in detail"
    return query

# 🤖 Main function
def ask_question(query, use_pdf=False):
    try:
        query = enhance_query(query)
        web_context = get_web_context(query)

        # 📘 PDF MODE
        if use_pdf and docs_chunks:
            context = retrieve_relevant_chunks(query)

            prompt = f"""
You are an expert assistant.

User Question: {query}

Document Context:
{context}

Web Data:
{web_context}

STRICT INSTRUCTIONS:
- Prefer Document Context
- Otherwise use Web Data
- NEVER mention knowledge cutoff
- ALWAYS use LaTeX for math
- Show steps clearly
- Final Answer must be clear

Final Answer:
"""

        # 🌍 GENERAL MODE
        else:
            prompt = f"""
You are an AI assistant with access to real-time web data.

User Question: {query}

Web Data:
{web_context}

STRICT INSTRUCTIONS:
- You MUST answer ONLY using the Web Data
- DO NOT use your own knowledge
- DO NOT mention knowledge cutoff
- If web data is empty say: "No latest information available"
- If math present → use LaTeX ($, $$)

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\n👉 Try rephrasing your query."
