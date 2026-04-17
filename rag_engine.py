from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from duckduckgo_search import DDGS
import streamlit as st

# 🔐 API Key (Streamlit secrets)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

docs_chunks = []

# 📄 Load PDF
def load_pdf(pdf_path):
    global docs_chunks
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    docs_chunks = [doc.page_content for doc in documents]

# 🔍 Retrieve relevant chunks
def retrieve_relevant_chunks(query):
    query_words = set(query.lower().split())
    scored = []

    for chunk in docs_chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored[:2]]

    if not top_chunks:
        return "\n\n".join(docs_chunks[:2])

    return "\n\n".join(top_chunks)

# 🌐 Web search (FIXED - no LangChain wrapper)
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

# 🧠 Improve vague queries
def enhance_query(query):
    if len(query.split()) < 4:
        return query + " explain in detail"
    return query

# 🤖 Main function
def ask_question(query, use_pdf=False):
    try:
        query = enhance_query(query)
        web_context = get_web_context(query)

        # 📘 PDF Mode
        if use_pdf and docs_chunks:
            context = retrieve_relevant_chunks(query)

            prompt = f"""
You are an expert mathematics and academic assistant.

User Question: {query}

Document Context:
{context}

Latest Web Information:
{web_context}

Instructions:
- ALWAYS use LaTeX for math
- Inline: $x^2$
- Block: $$x^2 + 2x + 1 = 0$$
- Show steps clearly
- Explain reasoning
- Clearly mark Final Answer

Final Answer:
"""

        # 🌍 General Mode
        else:
            prompt = f"""
You are a helpful AI assistant with access to latest web data.

User Question: {query}

Latest Web Information:
{web_context}

Instructions:
- Use latest info
- If math present → use LaTeX ($, $$)
- Do NOT hallucinate

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
