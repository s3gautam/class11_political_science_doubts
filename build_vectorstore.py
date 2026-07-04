import os
import gc
import shutil

from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==================================
# CONFIG
# ==================================

PDF_PATH = "ncert_political_science.pdf"
VECTOR_DB_PATH = "vectorstore"

# Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\siddhant.g_perfios\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# Poppler
POPPLER_PATH = (
    r"C:\Users\siddhant.g_perfios\Downloads\Release-26.02.0-0\poppler\Library\bin"
)

# ==================================
# GET PAGE COUNT
# ==================================

print("Reading PDF information...")

pdf_info = pdfinfo_from_path(
    PDF_PATH,
    poppler_path=POPPLER_PATH
)

total_pages = pdf_info["Pages"]

print(f"Total pages: {total_pages}")

# ==================================
# OCR PAGE BY PAGE
# ==================================

documents = []

for page_num in range(1, total_pages + 1):

    print(f"OCR page {page_num}/{total_pages}")

    try:

        pages = convert_from_path(
            PDF_PATH,
            first_page=page_num,
            last_page=page_num,
            dpi=150,
            poppler_path=POPPLER_PATH
        )

        page = pages[0]

        text = pytesseract.image_to_string(
            page,
            lang="eng",
            config="--psm 6"
        )

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num
                    }
                )
            )

        del page
        del pages
        gc.collect()

    except Exception as e:
        print(f"Error on page {page_num}: {e}")

print(f"\nDocuments created: {len(documents)}")

if len(documents) == 0:
    raise ValueError(
        "No text extracted from PDF."
    )

# ==================================
# CHUNKING
# ==================================

print("\nSplitting text into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError(
        "No chunks generated."
    )

# ==================================
# EMBEDDINGS
# ==================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================
# DELETE OLD VECTOR STORE
# ==================================

if os.path.exists(VECTOR_DB_PATH):

    print("\nRemoving old vectorstore...")

    shutil.rmtree(VECTOR_DB_PATH)

# ==================================
# CREATE FAISS VECTOR STORE
# ==================================

print("\nCreating FAISS index...")

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local(VECTOR_DB_PATH)

# ==================================
# DONE
# ==================================

print("\n==================================")
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print(f"Location: {VECTOR_DB_PATH}")
print(f"Pages processed: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print("==================================")