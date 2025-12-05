import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import re
from collections import defaultdict

!pip install rouge-score nltk sentence-transformers scikit-learn

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if needed
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load the instruction following dataset
dataset_path = "/content/drive/MyDrive/Colab Notebooks/data/eval/instruction_samples_full.jsonl"
dataset = load_dataset(dataset_path)
print(f"Loaded {len(dataset)} instruction samples")
print(f"\nExample sample:")
print(json.dumps(dataset[0], indent=2))

# Configure your model path here
MODEL_PATH = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Update this with your actual model path

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir="/root/.cache/huggingface")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="/root/.cache/huggingface"
)
print("Model loaded successfully!")



def create_prompt(sample: Dict[str, Any]) -> str:
    """Create a prompt for instruction following."""
    prompt = sample['prompt']

    # Simple instruction format - adjust based on your model
    formatted_prompt = f"""Question: {prompt}

Answer:"""
    return formatted_prompt

def generate_response(prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
    """Generate response from the model.

    Args:
        prompt: The input prompt
        max_length: Maximum length of generated tokens
        temperature: Sampling temperature

    Returns:
        Generated response string
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from output
    if prompt in full_text:
        response = full_text[len(prompt):].strip()
    else:
        # Fallback: decode only new tokens
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return response

# Test response generation
test_sample = dataset[0]
test_prompt = create_prompt(test_sample)
print("Test prompt:")
print(test_prompt)
print("\nGenerating response...")
test_response = generate_response(test_prompt)
print("\nGenerated response:")
print(test_response)
print("\nExpected response:")
print(test_sample['expected_response'])

def calculate_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores.

    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure,
    }

# Test ROUGE
print("Testing ROUGE metric:")
test_rouge = calculate_rouge(test_response, test_sample['expected_response'])
for metric, score in test_rouge.items():
    print(f"{metric}: {score:.4f}")

# Load sentence transformer model for semantic similarity
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

def calculate_semantic_similarity(generated: str, reference: str) -> float:
    """Calculate semantic similarity using sentence embeddings.

    Returns cosine similarity between embeddings (0 to 1).
    """
    # Generate embeddings
    embeddings = embedding_model.encode([generated, reference])

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return float(similarity)

# Test semantic similarity
print("\nTesting Semantic Similarity metric:")
test_sim = calculate_semantic_similarity(test_response, test_sample['expected_response'])
print(f"Semantic similarity: {test_sim:.4f}")

def evaluate_instruction_following(dataset: List[Dict]) -> Dict[str, Any]:
    """Run complete instruction following evaluation.

    Args:
        dataset: List of instruction samples

    Returns:
        Dictionary containing evaluation results
    """
    results = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'semantic_similarity': [],
        'samples': []
    }

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        # Create prompt and generate response
        prompt = create_prompt(sample)
        generated = generate_response(prompt)
        reference = sample['expected_response']

        # Calculate all metrics
        rouge_scores = calculate_rouge(generated, reference)
        sem_sim = calculate_semantic_similarity(generated, reference)

        # Store metrics
        results['rouge1'].append(rouge_scores['rouge1_f'])
        results['rouge2'].append(rouge_scores['rouge2_f'])
        results['rougeL'].append(rouge_scores['rougeL_f'])
        results['semantic_similarity'].append(sem_sim)

        # Store sample results
        results['samples'].append({
            'idx': idx,
            'id': sample['id'],
            'prompt': sample['prompt'],
            'generated': generated,
            'reference': reference,
            'rouge1': rouge_scores['rouge1_f'],
            'rouge2': rouge_scores['rouge2_f'],
            'rougeL': rouge_scores['rougeL_f'],
            'semantic_similarity': sem_sim

        })

    # Calculate aggregate metrics
    results['aggregate'] = {
        'rouge1': np.mean(results['rouge1']),
        'rouge2': np.mean(results['rouge2']),
        'rougeL': np.mean(results['rougeL']),
        'semantic_similarity': np.mean(results['semantic_similarity'])

    }

    return results

eval_results = evaluate_instruction_following(dataset)

# Print aggregate results
print("="*60)
print("INSTRUCTION FOLLOWING EVALUATION RESULTS")
print("="*60)
print(f"\nDataset size: {len(dataset)} samples\n")

print("Text Overlap Metrics:")
print(f"  ROUGE-1: {eval_results['aggregate']['rouge1']:.4f}")
print(f"  ROUGE-2: {eval_results['aggregate']['rouge2']:.4f}")
print(f"  ROUGE-L: {eval_results['aggregate']['rougeL']:.4f}")
print(f"\nSemantic Understanding:")
print(f"  Semantic Similarity: {eval_results['aggregate']['semantic_similarity']:.4f}")