import os
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# 1. LOAD API KEY
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )

print("API key loaded: True")


# =========================================================
# 2. CONNECT TO GEMINI
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini connected successfully.")


# =========================================================
# 3. LOAD DATASETS
# =========================================================

sales_df = pd.read_csv(
    "datasets/sales.csv"
)

inventory_df = pd.read_csv(
    "datasets/inventory.csv"
)

reviews_df = pd.read_csv(
    "datasets/customer_reviews.csv"
)

print("\nDatasets loaded successfully.")
print("Sales rows:", len(sales_df))
print("Inventory rows:", len(inventory_df))
print("Review rows:", len(reviews_df))


# =========================================================
# 4. CREATE DOCUMENTS
# =========================================================

documents = []


# -------------------------
# Sales documents
# -------------------------

for _, row in sales_df.iterrows():

    text = "Sales Data: " + " | ".join(
        f"{column}: {row[column]}"
        for column in sales_df.columns
    )

    documents.append(text)


# -------------------------
# Inventory documents
# -------------------------

for _, row in inventory_df.iterrows():

    text = "Inventory Data: " + " | ".join(
        f"{column}: {row[column]}"
        for column in inventory_df.columns
    )

    documents.append(text)


# -------------------------
# Customer review documents
# -------------------------

for _, row in reviews_df.iterrows():

    text = "Customer Review: " + " | ".join(
        f"{column}: {row[column]}"
        for column in reviews_df.columns
    )

    documents.append(text)


print("Total documents:", len(documents))


# =========================================================
# 5. LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)

print("Embedding model loaded successfully.")


# =========================================================
# 6. CREATE EMBEDDINGS
# =========================================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings created successfully.")


# =========================================================
# 7. CREATE FAISS INDEX
# =========================================================

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

print("\nFAISS index created successfully.")
print("Number of vectors:", index.ntotal)


# =========================================================
# 8. RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_documents(question, k=5):

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for i in indices[0]:

        if i < len(documents):

            results.append(
                documents[i]
            )

    return results


# =========================================================
# 9. GENERATE AI ANSWER
# =========================================================

def generate_answer(question, retrieved_documents):

    context = "\n\n".join(
        retrieved_documents
    )

    prompt = f"""
You are an AI Enterprise Decision Intelligence assistant.

Use ONLY the retrieved business information to answer the user's question.

RETRIEVED BUSINESS INFORMATION:
{context}

USER QUESTION:
{question}

Give a SHORT, CLEAR and EASY-TO-UNDERSTAND business answer.

IMPORTANT RULES:

1. Use simple English.
2. Do not write long paragraphs.
3. Use bullet points.
4. Give only information relevant to the question.
5. Include important numbers when available.
6. Do not repeat information.
7. Do not invent information.
8. Do not include unnecessary confidence scores.
9. Do not mention the RAG process.
10. Give a practical business recommendation.
11. If the data is not sufficient, clearly say that.

Use exactly this format:

KEY FINDING:
- Main finding
- Supporting evidence
- Important number or fact if available

RECOMMENDATION:
- Practical business action
- Another action only if necessary

Do not add any other sections.
"""

    print("\nGenerating AI answer...")

    # =====================================================
    # TRY PRIMARY MODEL
    # =====================================================

    try:

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        if response and response.text:

            return response.text

    except Exception as e:

        print("\nPrimary Gemini model unavailable.")
        print("Trying fallback model...")


    # =====================================================
    # TRY FALLBACK MODEL
    # =====================================================

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text

    except Exception as e:

        print("\nFallback Gemini model unavailable.")
        print("Gemini error:", e)


    # =====================================================
    # IF BOTH MODELS FAIL
    # =====================================================

    return """
KEY FINDING:
- The business data was successfully retrieved.
- Gemini is temporarily unavailable, so an AI-generated summary could not be created.

RECOMMENDATION:
- Please try the question again after a short time.
"""


# =========================================================
# 10. TEST RAG PIPELINE
# =========================================================

if __name__ == "__main__":

    question = "Which products have low stock?"


    print("\n==========================================")
    print("USER QUESTION")
    print("==========================================")

    print(question)


    # -----------------------------------------------------
    # Retrieve relevant information
    # -----------------------------------------------------

    retrieved_documents = retrieve_documents(
        question,
        k=5
    )


    print("\n==========================================")
    print("RETRIEVED INFORMATION")
    print("==========================================")

    for document in retrieved_documents:

        print(document)

        print("------------------------------------------")


    # -----------------------------------------------------
    # Generate AI answer
    # -----------------------------------------------------

    answer = generate_answer(
        question,
        retrieved_documents
    )


    print("\n==========================================")
    print("AI ANSWER")
    print("==========================================")

    print(answer)


    print("\n==========================================")
    print("RAG PIPELINE COMPLETED")
    print("==========================================")