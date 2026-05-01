import os
import json
import requests

# ── config ────────────────────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434/api/generate"
MODEL_NAME          = "deepseek-v3.1:671b-cloud"
KNOWLEDGE_BASE_PATH = "app/Working_transcript/knowledge_base.json"

# ── load knowledge base once at module load ───────────────────────────────────
def _load_knowledge_base() -> dict:
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(
            f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}. "
            "Please create knowledge_base.json in Working_transcript folder."
        )
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

KNOWLEDGE_BASE = _load_knowledge_base()


# ── 1. detect if product question ─────────────────────────────────────────────
def is_product_question(text: str) -> bool:
    """
    Ask LLM: is this text asking about TaskMaster Pro?
    Returns True or False.
    """
    product_name = KNOWLEDGE_BASE.get("product_name", "the product")
    
    prompt = f"""You are a classifier. Your only job is to answer YES or NO.

Question: Is the following text asking about {product_name} — its pricing, features, plans, trial, support, training, security, or any product-related topic?

Text: "{text}"

Rules:
- Answer YES if it is a question about the product
- Answer NO if it is small talk, greetings, or unrelated
- Reply with ONLY YES or NO. Nothing else.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()["response"].strip().upper()
        return "YES" in result
        
    except Exception as e:
        raise RuntimeError(f"Ollama detection call failed: {str(e)}")


# ── 2. generate answer ────────────────────────────────────────────────────────
def generate_answer(text: str) -> str:
    """
    Generate answer from knowledge base using LLM.
    Returns natural language response.
    """
    prompt = f"""You are a knowledgeable assistant for {KNOWLEDGE_BASE.get("product_name", "the product")}.

Knowledge base:
{json.dumps(KNOWLEDGE_BASE, indent=2)}

Question: "{text}"

Your job:
- Find relevant info from knowledge base
- Answer in 2-4 clear sentences
- Use only info from knowledge base
- No bullet points, just natural conversation
- Answer directly without preamble
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()
        
    except Exception as e:
        raise RuntimeError(f"Ollama answer call failed: {str(e)}")


# ── 3. main function (calls 1 and 2) ──────────────────────────────────────────
def process_transcript(text: str) -> dict:
    """
    Main function called by endpoint.
    Returns: {"is_product_question": bool, "answer": str or None}
    """
    if not text or not text.strip():
        return {"frontend-text": text ,"is_product_question": False, "answer": None}

    # step 1: detect
    if not is_product_question(text):
        return {"frontend-text": text, "is_product_question": False, "answer": None}

    # step 2: generate answer
    answer = generate_answer(text)
    
    return {"frontend-text": text, "is_product_question": True, "answer": answer}