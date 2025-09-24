import os
import json
import requests
from typing import List, Dict, Any

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "qa_golden_dataset.json")

# Dummy token for testing - replace with a valid one if auth is enforced
# You can generate a temporary token by logging in and inspecting network requests
AUTH_TOKEN = os.getenv("EVAL_AUTH_TOKEN", "your_dummy_auth_token_here")

headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}

# --- Helper Functions ---

def load_golden_dataset() -> List[Dict[str, Any]]:
    """Loads the golden dataset from the JSON file."""
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Golden dataset not found at {GOLDEN_DATASET_PATH}")
        return []

def query_chat_api(question: str) -> Dict[str, Any]:
    """Queries the chat API and returns the JSON response."""
    payload = {
        "session_id": "evaluation_session",
        "message": question
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat/", json=payload, headers=headers, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return {"answer": "", "sources": []}

def evaluate_sources(retrieved_sources: List[Dict[str, Any]], golden_sources: List[Dict[str, Any]]) -> bool:
    """Checks if at least one golden source is present in the retrieved sources."""
    if not golden_sources:
        return True  # No specific source required
    
    retrieved_set = {(s.get('source', ''), s.get('page')) for s in retrieved_sources}
    golden_set = {(g.get('source', ''), g.get('page')) for g in golden_sources}
    
    return not golden_set.isdisjoint(retrieved_set)

# --- Main Evaluation Logic ---

def main():
    """Runs the full evaluation process."""
    dataset = load_golden_dataset()
    if not dataset:
        return

    results = []
    total_questions = len(dataset)
    correct_sources = 0

    print(f"Starting evaluation with {total_questions} questions...\n")

    for i, item in enumerate(dataset):
        question = item['question']
        golden_answer = item['golden_answer']
        golden_sources = item['golden_sources']

        print(f"[{i+1}/{total_questions}] Querying for: '{question}'")
        api_response = query_chat_api(question)
        
        ai_answer = api_response.get('answer', '')
        retrieved_sources = api_response.get('sources', [])

        # Evaluation
        source_is_correct = evaluate_sources(retrieved_sources, golden_sources)
        if source_is_correct:
            correct_sources += 1

        results.append({
            "question": question,
            "golden_answer": golden_answer,
            "ai_answer": ai_answer,
            "source_is_correct": source_is_correct,
            "retrieved_sources": retrieved_sources
        })
        
        print(f"  -> Source Correct: {'Yes' if source_is_correct else 'No'}")
        print(f"  -> AI Answer: {ai_answer[:100]}...\n")

    # --- Report Results ---
    print("\n--- Evaluation Report ---")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Sources Found: {correct_sources} ({correct_sources/total_questions:.2%})")
    print("-------------------------")

    # Optionally, save detailed results to a file
    with open("evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Detailed results saved to evaluation_results.json")

if __name__ == "__main__":
    main()

