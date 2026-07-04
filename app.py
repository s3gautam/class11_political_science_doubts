import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ==================================
# CONFIG
# ==================================

VECTOR_DB_PATH = "vectorstore"

# Replace with your Gemini API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ==================================
# RESPONSE PARSER
# ==================================

def extract_answer(response):
    try:
        content = getattr(response, "content", response)

        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []

            for item in content:

                if item is None:
                    continue

                if isinstance(item, str):
                    parts.append(item)

                elif hasattr(item, "text"):
                    parts.append(str(item.text))

                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    else:
                        parts.append(str(item))

                else:
                    parts.append(str(item))

            return "\n".join(parts).strip()

        if isinstance(content, dict):

            if "text" in content:
                return str(content["text"]).strip()

            return str(content).strip()

        return str(content).strip()

    except Exception as e:
        return f"Error extracting response: {e}"


# ==================================
# LOAD EMBEDDINGS
# ==================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==================================
# LOAD VECTORSTORE
# ==================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ==================================
# LOAD LLM
# ==================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )


db = load_vectorstore()
llm = load_llm()

# ==================================
# UI
# ==================================

st.set_page_config(
    page_title="NCERT Political Science QA",
    layout="centered"
)

st.title("NCERT Political Science Q&A")

question = st.text_input(
    "Question"
)

marks = st.selectbox(
    "Marks",
    [
        "Not Specified",
        "1",
        "2",
        "3",
        "4",
        "5"
    ]
)

# ==================================
# GENERATE
# ==================================

if st.button("Generate Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Generating answer..."):

        docs = db.similarity_search(
            question,
            k=4
        )

        if not docs:
            st.error("No relevant content found.")
            st.stop()

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        page_no = docs[0].metadata.get(
            "page",
            "Unknown"
        )

        if marks == "Not Specified":

            answer_instruction = """
Give a complete and concise answer.

Return only the answer.

Do not write:
- Answer:
- Introduction:
- Conclusion:
- Notes:
- References:
- Any extra commentary
"""

        else:

            answer_instruction = f"""
The answer must be suitable for a {marks}-mark examination question.

Length guide:

1 mark:
20-30 words

2 marks:
40-60 words

3 marks:
80-100 words

4 marks:
100-140 words

5 marks:
150-200 words

Return only the answer.

Do not write:
- Answer:
- Introduction:
- Conclusion:
- Notes:
- References:
- Any extra commentary
"""

        prompt = f"""
You are an NCERT Political Science teacher.

Use ONLY the textbook context provided below.

Rules:
- Do not invent facts.
- Use only information from the context.
- Write in simple exam-friendly language.
- If the answer is unavailable, respond exactly:
I could not find the answer in the textbook.

{answer_instruction}

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        answer = extract_answer(response)

        st.text_area(
            "Answer",
            value=answer,
            height=300,
            disabled=True
        )

        st.text_input(
            "Page Number",
            value=str(page_no),
            disabled=True
        )