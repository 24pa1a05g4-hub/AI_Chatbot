import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
    
    if INDEX_NAME not in pc.list_indexes().names():
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

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    context_sources: List[str]

def get_embedding(input_text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Generate embeddings for the input text."""
    # Sanitize text for embedding
    sanitized_text = input_text.replace("\\n", " ")
    response = openai_client.embeddings.create(input=[sanitized_text], model=model)
    return response.data[0].embedding

def get_answer_from_openai(question: str, context: str) -> str:
    """Generate an answer using OpenAI's model based on the context."""
    prompt = f"Answer the question based only on the following context:\\n\\nContext: {context}\\n\\nQuestion: {question}\\n\\nAnswer:"
    response = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct", # A suitable model for completion
        prompt=prompt,
        max_tokens=150,
        temperature=0.1
    )
    return response.choices[0].text.strip()

@app.post("/ask", response_model=QueryResponse, operation_id="ask_question_from_data")
async def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # 1. Generate embedding for the query
    query_embedding = get_embedding(request.query)

    # 2. Search Pinecone for relevant documents
    # Adjust top_k and other parameters as needed for your index
    search_results = pinecone_index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    context_list = []
    for match in search_results.get("matches", []):
        if match.get("score", 0) > 0.7: # Filter by a relevance score
            content_text = match.metadata.get("text", "") # Ensure your metadata key matches
            if content_text:
                context_list.append(content_text)
    
    if not context_list:
        return QueryResponse(query=request.query, answer="No relevant information found in the data.", context_sources=[])

    # 3. Use OpenAI to generate a final answer based on context
    context_str = " ".join(context_list)
    answer = get_answer_from_openai(request.query, context_str)
    
    return QueryResponse(
        query=request.query,
        answer=answer,
        context_sources=context_list
    )

@app.get("/")
def read_root():
    return {"message": "RAG FastAPI service is running. Visit /docs to use the API."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
