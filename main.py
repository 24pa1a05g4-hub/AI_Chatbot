import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- 1. NEW IMPORT
from pydantic import BaseModel
from typing import List

from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Initialize clients using environment variables
try:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
    
    # Check if index exists safely
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        raise ValueError(f"Index '{INDEX_NAME}' not found.")
        
    pinecone_index = pc.Index(INDEX_NAME)

except KeyError as e:
    raise RuntimeError(f"Missing environment variable: {e}")
except ValueError as e:
    raise RuntimeError(e)

app = FastAPI(
    title="FastAPI OpenAI Pinecone RAG App",
    description="A RAG server for document search and question answering"
)

# --- 2. ADD CORS MIDDLEWARE (CRITICAL FIX) ---
# This allows your website (frontend) to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows ALL websites. Safe for testing. In production, change "*" to "https://yourwebsite.com"
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (POST, GET, etc.)
    allow_headers=["*"], # Allows all headers
)
# ---------------------------------------------

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    context_sources: List[str]

def get_embedding(input_text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embeddings for the input text."""
    # Sanitize text for embedding
    sanitized_text = input_text.replace("\n", " ")
    # Using 'text-embedding-3-small' is cheaper and better than ada-002
    response = openai_client.embeddings.create(input=[sanitized_text], model=model)
    return response.data[0].embedding

def get_answer_from_openai(question: str, context: str) -> str:
    """Generate an answer using OpenAI's chat model."""
    
    # Strict system prompt to prevent hallucination
    system_instruction = (
        "You are a helpful assistant. Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say 'I do not have that information based on the provided data.'"
    )

    # We use chat.completions.create (Modern API)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", # You can also use "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        max_tokens=250,
        temperature=0.1 
    )
    return response.choices[0].message.content.strip()

# --- 3. RENAMED ENDPOINT TO /chat TO MATCH FRONTEND ---
@app.post("/chat", response_model=QueryResponse, operation_id="ask_question_from_data")
async def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # 1. Generate embedding for the query
    try:
        query_embedding = get_embedding(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Embedding Error: {str(e)}")

    # 2. Search Pinecone
    try:
        search_results = pinecone_index.query(vector=query_embedding, top_k=3, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone Search Error: {str(e)}")
    
    context_list = []
    for match in search_results.get("matches", []):
        if match.get("score", 0) > 0.6: # Score threshold
            content_text = match.metadata.get("text", "")
            if content_text:
                context_list.append(content_text)
    
    # If no relevant context found in Pinecone
    if not context_list:
        return QueryResponse(
            query=request.query, 
            answer="I do not have that information based on the provided data.", 
            context_sources=[]
        )

    # 3. Generate Answer
    context_str = "\n\n".join(context_list)
    try:
        answer = get_answer_from_openai(request.query, context_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Generation Error: {str(e)}")
    
    return QueryResponse(
        query=request.query,
        answer=answer,
        context_sources=context_list
    )

@app.get("/")
def read_root():
    return {"message": "RAG Service Running. Send POST requests to /chat"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)