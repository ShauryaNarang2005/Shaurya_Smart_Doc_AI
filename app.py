import streamlit as st
from rag_engine import load_pdf, ask_question
import tempfile
import re

st.set_page_config(page_title="SmartDoc AI", layout="centered")

st.title("🤖 SmartDoc AI")

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload a PDF (optional)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    load_pdf(temp_path)
    st.success("✅ PDF Loaded Successfully!")

# 💬 Input
query = st.text_input("Ask anything...")

if query:
    with st.spinner("Thinking..."):
        answer = ask_question(query, use_pdf=uploaded_file is not None)

    st.markdown("## 🤖 Answer:")
    st.markdown("---")

    # ✅ Better LaTeX rendering (cleaner)
    parts = re.split(r'(\$\$.*?\$\$|\$.*?\$)', answer, flags=re.DOTALL)

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # Block LaTeX
        if part.startswith('$$') and part.endswith('$$'):
            try:
                st.latex(part[2:-2].strip())
            except:
                st.markdown(part)

        # Inline LaTeX
        elif part.startswith('$') and part.endswith('$'):
            try:
                st.latex(part[1:-1].strip())
            except:
                st.markdown(part)

        else:
            st.markdown(part)