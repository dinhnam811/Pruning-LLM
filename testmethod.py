import os
import time
import torch
import subprocess
import re

# =============================================================
# 1) YOUR HELPER FUNCTIONS (UNCHANGED)
# =============================================================

def generate_code(model, tokenizer, task):
    """
    Generate Java code for a given task using a model.
    Returns: (generated_code, time_taken_in_seconds)
    """
    signature = task["signature"]

    # Create prompt
    prompt = f"""
    You are a strict Java code generator.
You MUST follow these unbreakable rules exactly:

1. Output EXACTLY ONE Java method.
2. ZERO explanations.
3. ZERO repeated methods.
4. ZERO comments.
5. ZERO blank copies of the method.
6. DO NOT output '// Your code here'.
7. Output MUST start with 'public static'.
8. Output MUST end with the closing bracket }} of that method.
9. NO PLACEHOLDER METHODS.
10. Output ONLY the final implementation of the method
11. DO NOT output any placeholder, template, or empty method body.
12.DO NOT output any method skeleton.
13. DO NOT output the method twice.
Task:
{task['prompt']}

Signature: {signature}

Write the complete method:"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate and measure time
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed_time = time.time() - start_time

    # Decode only the new tokens
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    code = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Clean up the output
    code = code.replace("```java", "").replace("```", "").strip()

    # Extract just the method
    if "public static" in code:
        code = code[code.index("public static"):]
        if "}" in code:
            code = code[:code.rfind("}")+1]

    return code, elapsed_time


def test_code(task, method_code):
    """
    Test if the generated code passes all test cases.
    Returns: True if all tests pass, False otherwise
    """
    try:
        # Parse signature to get return type and method name
        sig_match = re.search(r'public\s+static\s+(\S+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', task["signature"])
        if not sig_match:
            return False

        return_type, method_name = sig_match.group(1), sig_match.group(2)

        # Build test code
        test_calls = []
        for i, test in enumerate(task["tests"], 1):
            inp, expected = test["input"], test["expected"]

            if return_type == "String":
                condition = f"!{method_name}({inp}).equals({expected})"
            else:
                condition = f"{method_name}({inp}) != {expected}"

            test_calls.append(f"if ({condition}) throw new Exception(\"Test {i} failed\");")

        # Create full Java file
        java_code = f"""
public class Main {{
    {method_code}

    public static void main(String[] args) {{
        try {{
{chr(10).join('            ' + tc for tc in test_calls)}
            System.out.println("OK");
        }} catch (Exception e) {{
            System.out.println("FAIL");
        }}
    }}
}}
"""

        # Write to file
        os.makedirs("temp", exist_ok=True)
        with open("temp/Main.java", "w") as f:
            f.write(java_code)

        # Compile
        compile_result = subprocess.run(["javac", "temp/Main.java"], capture_output=True, text=True)
        if compile_result.returncode != 0:
            return False

        # Run
        run_result = subprocess.run(["java", "-cp", "temp", "Main"], capture_output=True, text=True, timeout=5)
        return run_result.stdout.strip() == "OK"

    except Exception:
        return False

print("Helper functions ready!")

# =============================================================
# 2) SAMPLE PROBLEM (YOUR PROVIDED TASK)
# =============================================================

sample_task = {
    "id": "java_001_is_prime",
    "prompt": "Write a Java method isPrime that returns true if n is a prime number, otherwise false.",
    "signature": "public static boolean isPrime(int n)",
    "tests": [
        {"input": "2", "expected": "true"},
        {"input": "4", "expected": "false"},
        {"input": "17", "expected": "true"}
    ]
}

# =============================================================
# 3) LOAD YOUR MODEL + TOKENIZER
# =============================================================

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "C:/Users/namnd/Documents/QwenCoder-50"   # <-- change this

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print("Model loaded on:", next(model.parameters()).device)

# =============================================================
# 4) RUN GENERATION + TEST
# =============================================================

print("\nGenerating code for:", sample_task["id"])
generated_code, gen_time = generate_code(model, tokenizer, sample_task)

print("\n===== GENERATED CODE =====")
print(generated_code)

print(f"\nGeneration time: {gen_time:.2f} sec")

print("\nRunning tests...")
passed = test_code(sample_task, generated_code)

print("\n===== RESULT =====")
print("PASS ✔" if passed else "FAIL ✘")
