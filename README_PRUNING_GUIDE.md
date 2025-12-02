# LLM Pruning Guide: A Complete Beginner's Explanation

## Table of Contents
1. [What is an LLM?](#what-is-an-llm)
2. [What is Model Pruning?](#what-is-model-pruning)
3. [The Wanda Pruning Technique](#the-wanda-pruning-technique)
4. [Understanding Model Layers](#understanding-model-layers)
5. [The Complete Pruning Process](#the-complete-pruning-process)
6. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## What is an LLM?

### Simple Explanation
An **LLM (Large Language Model)** is like a very sophisticated autocomplete system. Think of it like the suggestion feature on your phone's keyboard, but millions of times more powerful.

### How It Works (Simple Analogy)
Imagine a massive library with billions of books. The LLM has "read" all these books and learned patterns about how words and sentences fit together. When you ask it a question or give it a prompt, it uses these patterns to predict what words should come next, one word at a time.

### What's Inside an LLM?
An LLM is made up of millions (or billions) of **parameters** - think of these as tiny adjustable knobs. Each knob has a specific setting (a number) that helps the model make decisions about what word to generate next.

**Example:**
- Small models: 3 billion parameters (3,000,000,000 knobs)
- Large models: 70+ billion parameters (70,000,000,000+ knobs)

The model we're working with: **Qwen2.5-Coder-3B-Instruct** has 3 billion parameters and is specifically trained to understand and generate programming code.

---

## What is Model Pruning?

### The Problem
Having billions of parameters means:
- ❌ The model takes up a lot of storage space (several gigabytes)
- ❌ It requires powerful computers (expensive GPUs) to run
- ❌ It runs slowly on regular computers
- ❌ It uses a lot of electricity

### The Solution: Pruning
**Pruning** means removing the least important parameters to make the model smaller and faster, while trying to keep it just as smart.

### Garden Analogy
Think of pruning like trimming a tree:
- 🌳 A tree has thousands of branches and leaves
- ✂️ A gardener cuts away dead branches and excess leaves
- 🌿 The tree stays healthy but becomes more manageable
- ⚡ The tree can now focus its energy on the important branches

Similarly, we identify which parameters in the LLM are doing very little work and remove them. The model becomes:
- ✅ Smaller (takes up less storage)
- ✅ Faster (runs quicker)
- ✅ More efficient (uses less memory and power)
- ✅ Still intelligent (if done carefully)

### Our Goal
In this project, we're pruning **5% of the parameters** (removing 150 million out of 3 billion). This means we keep 95% of the model while making it more efficient.

---

## The Wanda Pruning Technique

### What is Wanda?
**Wanda** stands for "Pruning by Weights AND Activations." It's a smart way to decide which parameters to remove.

### How It Works

#### Step 1: Understanding Importance
Not all parameters are equally important. Wanda uses two factors to determine importance:

1. **Weight Magnitude** (How big the number is)
   - A parameter with value 5.2 is more important than one with 0.001
   - Like a volume knob: turning a knob at 10 makes more difference than one at 0.1

2. **Activation** (How often it's actually used)
   - Some parameters are used frequently when processing text
   - Others rarely activate
   - Like apps on your phone: some you use daily, others sit unused

#### Step 2: The Formula
```
Importance = |Weight| × Activation
```

This means:
- **High Weight + High Activation = Very Important** (Don't prune!)
- **Low Weight + Low Activation = Not Important** (Safe to prune)
- **High Weight + Low Activation = Moderate** (Maybe prune)

#### Step 3: Selective Removal
After calculating importance for all parameters:
1. Sort them from most to least important
2. Keep the top 95% most important ones
3. Set the bottom 5% to zero (remove them)

### Why Wanda is Smart
Other pruning methods only look at weight size OR usage. Wanda looks at BOTH, making better decisions about what to keep.

**Example:**
- ❌ A parameter with weight 10.0 but never used → Looks important but isn't
- ❌ A parameter with weight 0.5 but used constantly → Looks unimportant but is
- ✅ Wanda catches both cases correctly!

---

## Understanding Model Layers

An LLM is organized in **layers**, like a skyscraper with many floors. Our model has multiple layers, and each layer contains several types of components.

### The Two Main Components We're Pruning

#### 1. Attention Mechanism (The "Focus System")
This is how the model decides which words to pay attention to when generating the next word.

**Real-Life Analogy:**
Imagine you're reading this sentence: "The cat sat on the mat because **it** was tired."
- What does "it" refer to? The cat or the mat?
- Your brain automatically looks back at "cat" because it makes sense
- The attention mechanism does the same thing for the LLM

**Four Types of Attention Layers:**

##### a) **Q_PROJ (Query Projection)**
- **What it does:** Creates "questions" about what the model is looking for
- **Analogy:** Like you asking "What am I trying to understand in this sentence?"
- **Technical:** Transforms input into query vectors

##### b) **K_PROJ (Key Projection)**
- **What it does:** Creates "labels" for each word in the context
- **Analogy:** Like labeling each word with what it represents (noun, verb, subject, etc.)
- **Technical:** Transforms input into key vectors that queries will match against

##### c) **V_PROJ (Value Projection)**
- **What it does:** Stores the actual "meaning" of each word
- **Analogy:** Like a dictionary that holds the definition once you find the right word
- **Technical:** Transforms input into value vectors that carry the information

##### d) **O_PROJ (Output Projection)**
- **What it does:** Combines all the attention results into a final answer
- **Analogy:** Like summarizing all the clues you found into one coherent understanding
- **Technical:** Projects the attention output back to the model dimension

**How They Work Together:**
1. **Q** asks: "What should I focus on?"
2. **K** responds: "Here's what each word is about"
3. **Q** and **K** compare: "This word matches what I'm looking for!"
4. **V** provides: "Here's the meaning of that word"
5. **O** summarizes: "Based on everything, here's the understanding"

#### 2. MLP Layers (The "Thinking System")
MLP stands for "Multi-Layer Perceptron" - a fancy term for the parts that do the actual "thinking" and transformation of information.

**Two Types of MLP Layers:**

##### a) **UP_PROJ (Up Projection)**
- **What it does:** Expands information into a larger space for processing
- **Analogy:** Like spreading out papers on a big desk so you can see everything clearly and make connections
- **Technical:** Increases the dimensionality to allow complex transformations
- **Example:** Takes 3000 numbers and expands them to 12,000 numbers

##### b) **DOWN_PROJ (Down Projection)**
- **What it does:** Compresses the processed information back down
- **Analogy:** Like writing a summary after reading all those papers - condensing the important insights
- **Technical:** Reduces dimensionality back to the model size
- **Example:** Takes 12,000 numbers and compresses them back to 3000 numbers

**How They Work Together:**
1. **UP_PROJ:** "Let me expand this information to find patterns"
2. (Middle processing happens here with activation functions)
3. **DOWN_PROJ:** "Now let me compress the insights back into a useful form"

### Visualization of One Layer

```
Input Text: "Write a Java function"
         ↓
┌────────────────────────────────────┐
│     ATTENTION MECHANISM            │
│  ┌──────────────────────────────┐  │
│  │ Q_PROJ (What am I looking for?)│  │
│  │ K_PROJ (What is each word?)   │  │
│  │ V_PROJ (Word meanings)        │  │
│  │ O_PROJ (Combine results)      │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│     MLP (THINKING)                 │
│  ┌──────────────────────────────┐  │
│  │ UP_PROJ (Expand for thinking) │  │
│  │ Processing...                 │  │
│  │ DOWN_PROJ (Compress results)  │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         ↓
Output: Generates next word
```

This pattern repeats across many layers (typically 24-48 layers in our model).

### Why We Prune These Specific Layers

These six layer types (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj) contain the vast majority of the model's parameters. By pruning just these layers, we can:
- ✅ Remove millions of parameters
- ✅ Keep the model's core structure intact
- ✅ Maintain good performance

We DON'T prune:
- Embedding layers (how words are converted to numbers)
- Layer normalization (keeps values stable)
- Final output layer (produces the final text)

---

## The Complete Pruning Process

### Overview
Our pruning process has 4 main stages:

```
1. LOAD → 2. CALIBRATE → 3. PRUNE → 4. SAVE
```

Let's break down each stage:

---

### Stage 1: Load Model & Data

#### What Happens:
1. Load the Qwen2.5-Coder-3B-Instruct model (3 billion parameters)
2. Load the tokenizer (converts text ↔ numbers)
3. Load Java code samples for calibration

#### Why Java Code?
- This model is designed for coding
- We want to prune it specifically for Java programming tasks
- By showing it Java code during calibration, we ensure it keeps parameters important for Java

#### Configuration:
- **Model:** Qwen2.5-Coder-3B-Instruct
- **Sparsity:** 5% (we'll remove 5% of parameters)
- **Calibration samples:** 80 Java code examples
- **Data source:** Small dataset of Java functions from real projects

---

### Stage 2: Collect Activations (Calibration)

#### What Happens:
We run 80 Java code samples through the model and watch which parameters activate (get used).

#### The Process:

1. **Hook Installation:**
   - Like installing sensors on each layer
   - These sensors measure how much each parameter activates
   - They don't change anything, just observe

2. **Feed Data:**
   - Send all 80 Java code samples through the model
   - For each sample, the sensors record which parameters activate strongly
   - Each sample might be something like:
   ```java
   public boolean isEven(int number) {
       return number % 2 == 0;
   }
   ```

3. **Calculate Average Activations:**
   - After processing all 80 samples, calculate the average activation for each parameter
   - Parameters used frequently get high activation scores
   - Parameters rarely used get low activation scores

#### Example:
```
Parameter #12,458,392:
  - Sample 1: Activation = 0.8
  - Sample 2: Activation = 0.9
  - Sample 3: Activation = 0.1
  ...
  - Sample 80: Activation = 0.7
  → Average Activation = 0.65 (fairly important!)
```

4. **Remove Hooks:**
   - After collecting all data, remove the sensors
   - The model is back to normal, but we now have activation data

#### Why This Matters:
Without this step, we'd only know the **weight sizes** but not which parameters are **actually used** for Java code. This calibration ensures we keep Java-relevant parameters.

---

### Stage 3: Prune the Model (Wanda Algorithm)

This is the main event where we actually remove parameters!

#### For Each Layer:

##### Step 3.1: Calculate Importance
```
For each parameter in the layer:
  Importance = |Weight| × Activation
```

**Example:**
- Parameter A: Weight = 2.5, Activation = 0.8 → Importance = 2.0
- Parameter B: Weight = 0.1, Activation = 0.3 → Importance = 0.03
- Parameter C: Weight = 5.0, Activation = 0.05 → Importance = 0.25

##### Step 3.2: Find Threshold
1. Flatten all importance scores into one long list
2. Sort them from highest to lowest
3. Find the value at the 95th percentile (this is our threshold)

**Example with 1000 parameters:**
```
Sorted importance: [5.2, 4.8, 4.1, ..., 0.8, 0.7, 0.6, ..., 0.01, 0.005, 0.001]
                    ↑                         ↑                                ↑
                  Rank 1                   Rank 950                       Rank 1000
                                             ↑
                                    Threshold (keep everything above)
```

##### Step 3.3: Create Mask
```
For each parameter:
  If importance >= threshold:
    mask = 1 (KEEP)
  Else:
    mask = 0 (REMOVE)
```

##### Step 3.4: Apply Mask
```
New Weight = Old Weight × Mask

Examples:
- Important parameter: 2.5 × 1 = 2.5 (kept)
- Unimportant parameter: 0.1 × 0 = 0.0 (pruned)
```

#### Statistics Collected:
- Total parameters examined
- Number of parameters pruned
- Actual sparsity achieved
- Per-layer breakdown

#### The Magic:
By setting unimportant parameters to exactly 0.0:
- The model still works
- Those parameters take up less space
- They can be skipped during computation (faster inference)
- The model maintains ~95% of its original capability

---

### Stage 4: Save & Validate

#### What Happens:

1. **Save Pruned Model:**
   - Save to directory: `./QwenCoder3B_JavaPruned-10`
   - Includes model weights (with 5% set to zero)
   - Includes tokenizer
   - Uses safe serialization format

2. **Quick Validation Test:**
   - Run a simple prompt through the pruned model
   - Example: "Write a Java function that returns true if a number is even."
   - Check that it generates coherent code (not gibberish or repetition)

3. **Success Indicators:**
   - ✅ Model generates sensible Java code
   - ✅ No repetitive output (like "hellohellohello")
   - ✅ File size is smaller than original
   - ✅ Sparsity target achieved (~5%)

---

## Step-by-Step Code Walkthrough

This section provides an **EXTREMELY DETAILED** line-by-line explanation of every single line in the pruning script. Even if you have no programming experience, you'll understand exactly what's happening.

---

### SECTION 1: Import Libraries (Lines 1-4)

#### **Line 1:** `import torch`

**What it does:**
- Imports the PyTorch library
- PyTorch is like a toolbox for working with neural networks and numbers

**Analogy:** Like bringing a toolbox into your workshop before starting a project

**What is PyTorch?**
- A Python library (collection of pre-written code) for machine learning
- Provides tools to work with large arrays of numbers (tensors)
- Can run calculations on GPUs (fast graphics cards) for speed

---

#### **Line 2:** `from transformers import AutoModelForCausalLM, AutoTokenizer`

**What it does:**
- Imports two specific tools from the Hugging Face Transformers library
- These tools help us load and work with language models

**Breaking it down:**
- `from transformers` = From the transformers library
- `import` = Bring in these specific tools
- `AutoModelForCausalLM` = A tool that can automatically load any language model
- `AutoTokenizer` = A tool that converts text to numbers and back

**Analogy:** Like taking specific tools (a hammer and screwdriver) from your toolbox instead of carrying the whole toolbox

**What is a tokenizer?**
- Converts human text into numbers the model can understand
- Example: "Hello world" → [15339, 1917]
- Also converts numbers back to text

**What is CausalLM?**
- "Causal Language Model" = A model that predicts the next word based on previous words
- Like autocomplete on your phone, but much more sophisticated

---

#### **Line 3:** `from datasets import load_dataset`

**What it does:**
- Imports a function to load datasets (collections of data)
- The `load_dataset` function can read data from files

**Analogy:** Like importing a tool that helps you open and read different types of files (PDF, Word, etc.)

---

#### **Line 4:** `from tqdm import tqdm`

**What it does:**
- Imports a progress bar tool
- Shows you how far along a process is (like a loading bar)

**Example output:**
```
Processing: 100%|████████████| 80/80 [02:15<00:00,  1.69s/it]
```

**Analogy:** Like the progress bar when you're downloading a file or installing software

---

#### **Line 5:** (Empty line)

**What it does:**
- Blank line for visual separation
- Makes code easier to read by grouping related sections

---

### SECTION 2: Configuration (Lines 6-10)

#### **Line 6:** `# Configuration`

**What it does:**
- This is a comment (not executed code)
- The `#` symbol tells Python "ignore this line, it's just a note for humans"
- Helps organize the code into sections

**Analogy:** Like a sticky note on a page explaining what the next section is about

---

#### **Line 7:** `MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"`

**What it does:**
- Creates a variable named `MODEL_NAME`
- Stores the name/ID of the model we want to use

**Breaking it down:**
- `MODEL_NAME` = Variable name (a container with a label)
- `=` = Assignment operator (puts something in the container)
- `"Qwen/Qwen2.5-Coder-3B-Instruct"` = The actual model identifier

**What is this model?**
- **Qwen** = Model family created by Alibaba Cloud
- **2.5-Coder** = Version 2.5, specialized for coding
- **3B** = 3 billion parameters
- **Instruct** = Fine-tuned to follow instructions

**Analogy:** Like writing "Toyota Camry 2024" on a note before going to the car dealership

---

#### **Line 8:** `SPARSITY = 0.05  # 5% pruning`

**What it does:**
- Sets how much of the model to prune (remove)
- 0.05 = 5% (we'll remove 5% of parameters)

**Breaking it down:**
- `SPARSITY` = Variable name
- `=` = Assignment
- `0.05` = The value (5% expressed as a decimal: 5/100 = 0.05)
- `# 5% pruning` = Comment explaining what the number means

**Math:**
- 0.05 = 5%
- If model has 3,000,000,000 parameters
- We'll prune: 3,000,000,000 × 0.05 = 150,000,000 parameters
- We'll keep: 3,000,000,000 × 0.95 = 2,850,000,000 parameters

**Analogy:** Like deciding to declutter your closet by removing 5% of your clothes

---

#### **Line 9:** `CALIBRATION_SIZE = 80`

**What it does:**
- Sets how many code samples to use for calibration
- We'll use 80 Java code examples to measure parameter importance

**Breaking it down:**
- `CALIBRATION_SIZE` = Variable name
- `=` = Assignment
- `80` = Number of samples

**Why 80?**
- More samples = more accurate measurement of what's important
- Too many samples = takes longer to process
- 80 is a good balance between accuracy and speed

**Analogy:** Like taste-testing 80 different dishes to learn someone's food preferences before planning a menu

---

#### **Line 10:** `SAVE_DIR = "./QwenCoder3B_JavaPruned-5"`

**What it does:**
- Sets the folder name where the pruned model will be saved

**Breaking it down:**
- `SAVE_DIR` = Variable name (DIR = directory = folder)
- `=` = Assignment
- `"./QwenCoder3B_JavaPruned-5"` = Folder path

**Path breakdown:**
- `./` = Current directory (the folder you're working in)
- `QwenCoder3B` = Model name
- `JavaPruned` = Indicates it's pruned using Java data
- `-5` = Indicates 5% sparsity

**Analogy:** Like writing "Pruned_Photos_2024" on a new folder before organizing photos

---

#### **Line 11:** (Empty line)

**What it does:**
- Visual separator between configuration and model loading sections

---

### SECTION 3: Load Model (Lines 12-16)

#### **Line 12:** `# Load Model`

**What it does:**
- Comment marking the start of the model loading section

---

#### **Line 13:** `print("Loading model...")`

**What it does:**
- Displays text to the screen to inform the user what's happening
- `print()` = Function that outputs text
- `"Loading model..."` = The message to display

**Output:**
```
Loading model...
```

**Why this is useful:**
- Loading can take 30-60 seconds
- User knows the program is working, not frozen

**Analogy:** Like a progress message when installing software: "Installing... please wait"

---

#### **Line 14:** `tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)`

**What it does:**
- Downloads and loads the tokenizer for our model
- Stores it in a variable called `tokenizer`

**Breaking it down:**
- `tokenizer` = Variable name (where we store the loaded tokenizer)
- `=` = Assignment
- `AutoTokenizer` = The tool that loads tokenizers
- `.from_pretrained()` = Method (function) that downloads the tokenizer
- `MODEL_NAME` = The model identifier we set earlier

**What happens:**
1. Checks if tokenizer is already downloaded
2. If not, downloads it from Hugging Face servers
3. Loads it into memory
4. Returns the tokenizer object

**File size:** ~2-5 MB

**What the tokenizer contains:**
- Vocabulary: List of 50,000+ tokens (words, subwords, symbols)
- Rules for splitting text into tokens
- Special tokens (start, end, padding, etc.)

**Example usage:**
```python
# Input text
text = "public boolean isEven(int n)"

# Tokenizer converts to numbers
tokens = [1898, 7778, 374, 23830, 1577, 308, 8]

# Can convert back to text
original = tokenizer.decode(tokens)  # "public boolean isEven(int n)"
```

**Analogy:** Like downloading a translation dictionary before traveling to a foreign country

---

#### **Line 15:** `model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")`

**What it does:**
- Downloads and loads the actual 3B parameter language model
- Configures it for efficient memory usage

**Breaking it down:**
- `model` = Variable where we store the loaded model
- `=` = Assignment
- `AutoModelForCausalLM` = Tool that loads causal language models
- `.from_pretrained()` = Method that downloads the model
- `MODEL_NAME` = Model identifier

**Parameters:**
1. `torch_dtype=torch.float16`
   - **What:** Sets the number precision to 16-bit floating point
   - **Why:** Uses half the memory (6GB instead of 12GB)
   - **Trade-off:** Tiny bit less precise, but usually not noticeable

2. `device_map="auto"`
   - **What:** Automatically decides where to put the model
   - **Options:** GPU (fast) or CPU (slower)
   - **Logic:** If GPU available, use it; otherwise use CPU

**What happens:**
1. Checks if model is already downloaded
2. If not, downloads ~6GB of model weights
3. Loads weights into memory
4. Puts model on GPU if available
5. Returns model object ready to use

**File size:** ~6 GB

**Download time:** 5-20 minutes (depending on internet speed)

**Analogy:** Like downloading a massive encyclopedia to your device so you can use it offline

---

#### **Line 16:** `model.eval()`

**What it does:**
- Puts the model into "evaluation mode"
- Disables certain training-specific features

**Breaking it down:**
- `model` = The model we just loaded
- `.eval()` = Method that switches to evaluation mode

**What changes in eval mode:**
1. **Dropout layers:** Turned off
   - Dropout randomly ignores some neurons during training
   - We want ALL neurons active when using the model

2. **Batch normalization:** Uses stored statistics
   - Ensures consistent behavior

3. **Gradient tracking:** Disabled (saves memory)

**Why this matters:**
- Makes model behavior consistent and deterministic
- Saves memory (we're not training, just observing)

**Analogy:** Like switching your phone from "charging mode" to "normal use mode"

---

#### **Line 17:** (Empty line)

**What it does:**
- Visual separator before data loading section

---

### SECTION 4: Load Calibration Data (Lines 18-26)

#### **Line 18:** `# Load Calibration Data`

**What it does:**
- Comment marking the data loading section

---

#### **Line 19:** `print("Loading Java calibration samples...")`

**What it does:**
- Informs user that data is being loaded

**Output:**
```
Loading Java calibration samples...
```

---

#### **Lines 20-24:** Dataset loading

```python
dataset = load_dataset("json", data_files={
    "train": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "validation": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",
    "test": "/content/drive/MyDrive/Colab Notebooks/data/test/test_small.jsonl"
})
```

**What it does:**
- Loads Java code samples from JSON files
- Creates a dataset object with train/validation/test splits

**Breaking it down line by line:**

**Line 20:** `dataset = load_dataset("json", data_files={`
- `dataset` = Variable to store the loaded data
- `load_dataset` = Function that loads data
- `"json"` = File format (JSON Lines)
- `data_files={` = Dictionary of file paths (opening brace)

**Line 21:** `"train": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",`
- `"train":` = Label for training data
- Path points to training dataset file
- `.jsonl` = JSON Lines format (one JSON object per line)

**Line 22:** `"validation": "/content/drive/MyDrive/Colab Notebooks/data/train/train_small.jsonl",`
- Same file as train (we're not training, just using for calibration)

**Line 23:** `"test": "/content/drive/MyDrive/Colab Notebooks/data/test/test_small.jsonl"`
- Points to test dataset file

**Line 24:** `})`
- `}` = Closing the dictionary
- `)` = Closing the function call

**Data format example:**
```json
{"repo": "ReactiveX/RxJava", "code": "public boolean isEmpty() { return size == 0; }", "docstring": "Checks if empty"}
```

Each line contains:
- `repo`: Where the code came from
- `code`: The actual Java code
- `docstring`: Description of what it does

**Analogy:** Like opening multiple recipe books and bookmarking specific pages you want to reference

---

#### **Line 25:** `texts = [ex["code"] for ex in dataset["train"].select(range(min(CALIBRATION_SIZE, len(dataset["train"]))))]`

**What it does:**
- Extracts just the code from the first 80 examples
- Creates a list of Java code strings

**Breaking it down (this is complex!):**

Let's read from inside out:

1. `len(dataset["train"])`
   - Gets total number of examples in training set
   - Example result: 1000

2. `min(CALIBRATION_SIZE, len(dataset["train"]))`
   - Takes smaller of: 80 or 1000
   - Result: 80
   - **Why:** If dataset has fewer than 80 examples, use all of them

3. `range(min(...))`
   - Creates sequence: 0, 1, 2, ..., 79
   - These are index numbers

4. `dataset["train"].select(range(...))`
   - Selects first 80 examples from training data
   - Like: dataset[0], dataset[1], ..., dataset[79]

5. `for ex in ...`
   - Loop through each selected example
   - `ex` = one example at a time

6. `ex["code"]`
   - Extract just the "code" field from each example
   - Ignores "repo" and "docstring" fields

7. `[... for ex in ...]`
   - List comprehension (compact way to build a list)
   - Collects all the code strings into one list

**Result:**
```python
texts = [
    "public boolean isEmpty() { return size == 0; }",
    "public int getSize() { return this.size; }",
    # ... 78 more Java code examples ...
]
```

**Analogy:** Like going through a cookbook, taking only the first 80 recipes, and writing down just the ingredients list (ignoring the photos and stories)

---

#### **Line 26:** `print(f"Using {len(texts)} calibration samples")`

**What it does:**
- Prints how many samples were loaded

**Breaking it down:**
- `print()` = Display text
- `f"..."` = f-string (formatted string) - allows embedding variables
- `{len(texts)}` = Insert the length of the texts list
- `len(texts)` = Count how many items in list (should be 80)

**Output:**
```
Using 80 calibration samples
```

**What is an f-string?**
- Modern Python way to insert variables into text
- `f"I have {5 + 5} apples"` → "I have 10 apples"

**Analogy:** Like printing a confirmation: "Successfully loaded 80 recipes"

---

#### **Line 27:** (Empty line)

**What it does:**
- Visual separator before function definitions

---

### SECTION 5: Collect Activations Function (Lines 28-73)

#### **Line 28:** `# Collect Layer Activations (Wanda Technique)`

**What it does:**
- Comment marking this section as the activation collection part
- Mentions this uses the Wanda technique

---

#### **Line 29:** `def collect_activations(model, tokenizer, texts):`

**What it does:**
- Defines a function named `collect_activations`
- This function will measure which parameters are important

**Breaking it down:**
- `def` = Define a function
- `collect_activations` = Function name
- `(model, tokenizer, texts)` = Inputs the function needs
  - `model`: The LLM we loaded
  - `tokenizer`: Converts text to numbers
  - `texts`: List of Java code samples

**What is a function?**
- Reusable piece of code
- Takes inputs, does work, returns output
- Like a recipe: takes ingredients, follows steps, produces a dish

**Analogy:** Like creating a recipe card that says: "To measure ingredient importance, you need: a dish (model), a measuring cup (tokenizer), and sample ingredients (texts)"

---

#### **Line 30:** `"""Collects activation scales for target Linear layers using Wanda technique."""`

**What it does:**
- Docstring (documentation string)
- Explains what the function does in human language

**Triple quotes (`"""`):**
- Used for multi-line strings
- Standard way to document functions in Python

---

#### **Line 31:** `torch.manual_seed(42)`

**What it does:**
- Sets the random number generator seed to 42
- Ensures reproducibility (same results every time)

**Breaking it down:**
- `torch` = PyTorch library
- `.manual_seed()` = Function to set random seed
- `42` = Arbitrary number (could be any number)

**Why this matters:**
- Some operations use randomness
- With same seed, randomness is "deterministic"
- You get identical results every time you run the code

**Example:**
```python
# Without seed:
random_number = random()  # Could be 0.7, then 0.3, then 0.9

# With seed(42):
random_number = random()  # Always 0.639...
```

**Why 42?**
- Reference to "Hitchhiker's Guide to the Galaxy" (the answer to everything)
- Common convention in programming examples
- Could be any number!

**Analogy:** Like setting your phone's clock to a specific time before taking photos, so all photos have consistent timestamps

---

#### **Line 32:** `target_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']`

**What it does:**
- Creates a list of layer names we want to prune
- These are the 6 important layer types in the model

**Breaking it down:**
- `target_keywords` = Variable name
- `=` = Assignment
- `[...]` = List (ordered collection)
- `'q_proj'`, etc. = Strings (text)

**The 6 layer types:**
1. `q_proj`: Query projection (attention - asks questions)
2. `k_proj`: Key projection (attention - provides labels)
3. `v_proj`: Value projection (attention - stores meanings)
4. `o_proj`: Output projection (attention - combines results)
5. `up_proj`: Up projection (MLP - expands for thinking)
6. `down_proj`: Down projection (MLP - compresses results)

**Why these specific layers?**
- They contain most of the model's parameters
- They're the "heavy lifters" of the model
- Other layers (embeddings, normalizations) are too critical to prune

**Analogy:** Like making a list of which parts of a car you can modify (suspension, exhaust, wheels) vs. which are critical (engine, brakes, steering)

---

#### **Line 33:** `target_modules = {}`

**What it does:**
- Creates an empty dictionary to store target modules
- Will be filled in the next loop

**Breaking it down:**
- `target_modules` = Variable name
- `=` = Assignment
- `{}` = Empty dictionary

**What is a dictionary?**
- Like a real dictionary: word → definition
- In Python: key → value
- Example: `{"apple": "red fruit", "banana": "yellow fruit"}`

**What will it store?**
- Key: layer name (string)
- Value: the actual layer object (module)
- Example: `{"model.layers.0.self_attn.q_proj": <Linear layer object>}`

**Analogy:** Like creating an empty phonebook that you'll fill with names and numbers

---

#### **Line 34:** `param_to_module = {}`

**What it does:**
- Creates another empty dictionary
- Maps parameter names to module names

**What will it store?**
- Key: parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
- Value: module name (e.g., "model.layers.0.self_attn.q_proj")

**Why two dictionaries?**
- `target_modules`: name → layer object (for installing hooks)
- `param_to_module`: parameter name → layer name (for matching activations to parameters)

**Analogy:** Like having two lists:
- List 1: Employee name → Employee object
- List 2: Email address → Employee name

---

#### **Line 35:** (Empty line)

**What it does:**
- Visual separator before the loop

---

#### **Line 36:** `# Find target layers`

**What it does:**
- Comment explaining the next section

---

#### **Line 37:** `for name, module in model.named_modules():`

**What it does:**
- Loop through every layer in the model
- Check each one to see if it's a target layer

**Breaking it down:**
- `for` = Start a loop
- `name, module` = Two variables: layer name and layer object
- `in` = Loop through
- `model.named_modules()` = Function that returns all layers with their names

**What is `named_modules()`?**
- Returns pairs: (name, module)
- Example pairs:
  - ("model", <Model object>)
  - ("model.layers", <LayerList object>)
  - ("model.layers.0", <Layer object>)
  - ("model.layers.0.self_attn", <Attention object>)
  - ("model.layers.0.self_attn.q_proj", <Linear object>) ← This is what we want!

**How many iterations?**
- A 3B model has hundreds or thousands of named modules
- Most we'll skip; only ~120 match our criteria

**Analogy:** Like going through every room in a large hotel, checking each room's name to see if it's a conference room

---

#### **Line 38:** `if isinstance(module, torch.nn.Linear) and any(k in name for k in target_keywords):`

**What it does:**
- Checks if this module is a Linear layer AND has one of our keywords in its name
- Only processes modules that pass both conditions

**Breaking it down:**

**Part 1:** `isinstance(module, torch.nn.Linear)`
- `isinstance()` = Check if object is a specific type
- `module` = The current layer
- `torch.nn.Linear` = Linear layer type
- Returns True if module is a Linear layer, False otherwise

**What is a Linear layer?**
- A layer that performs matrix multiplication: `output = input × weights + bias`
- Contains the bulk of model parameters
- Most important type for pruning

**Part 2:** `and`
- Logical AND operator
- Both conditions must be True

**Part 3:** `any(k in name for k in target_keywords)`
- `for k in target_keywords` = Loop through our 6 keywords
- `k in name` = Check if keyword is in the layer name
- `any(...)` = Return True if ANY keyword matches

**Example checks:**
```python
name = "model.layers.0.self_attn.q_proj"
"q_proj" in name → True ✓
isinstance(module, Linear) → True ✓
→ Process this module!

name = "model.layers.0.layer_norm"
No keywords match → False ✗
→ Skip this module
```

**Analogy:** Like filtering emails: only keep messages that are (from a specific sender) AND (contain certain keywords)

---

#### **Line 39:** `target_modules[name] = module`

**What it does:**
- Adds this module to our dictionary of target modules
- Key: layer name, Value: layer object

**Example:**
```python
target_modules["model.layers.0.self_attn.q_proj"] = <Linear layer object>
```

**Indentation:**
- This line is indented, meaning it's inside the `if` statement
- Only executes if the condition on line 38 is True

---

#### **Line 40:** `param_to_module[f"{name}.weight"] = name`

**What it does:**
- Creates a mapping from parameter name to module name
- Adds ".weight" to the name because we're tracking the weight parameter

**Breaking it down:**
- `param_to_module[...]` = Dictionary we're adding to
- `f"{name}.weight"` = Parameter name (module name + ".weight")
- `= name` = Maps to the module name

**Example:**
```python
param_to_module["model.layers.0.self_attn.q_proj.weight"] = "model.layers.0.self_attn.q_proj"
```

**Why ".weight"?**
- Each Linear layer has:
  - `weight`: The main parameter matrix (what we prune)
  - `bias`: Optional bias vector (usually not pruned)
- We only track weights

**Analogy:** Like creating a lookup table: "Parameter address" → "Which room it's in"

---

#### **Line 41:** (Empty line)

---

#### **Line 42:** `print(f"Targeting {len(target_modules)} layers for pruning")`

**What it does:**
- Prints how many target layers were found

**Breaking it down:**
- `len(target_modules)` = Count items in dictionary
- Should be around 120-150 layers for a typical model

**Output:**
```
Targeting 120 layers for pruning
```

**Why useful:**
- Confirms the search worked
- Sanity check: too few (< 50) might indicate a problem

---

#### **Line 43:** `activations = {name: [] for name in target_modules}`

**What it does:**
- Creates a dictionary to store activation measurements
- One empty list for each target module

**Breaking it down:**
- `activations` = Variable name
- `{... for name in target_modules}` = Dictionary comprehension
- `name: []` = Key: module name, Value: empty list

**Result:**
```python
activations = {
    "model.layers.0.self_attn.q_proj": [],
    "model.layers.0.self_attn.k_proj": [],
    # ... 120 more entries ...
}
```

**What will the lists store?**
- Each list will collect activation measurements from all 80 samples
- After processing: `activations["model.layers.0.self_attn.q_proj"]` = [tensor1, tensor2, ..., tensor80]

**Analogy:** Like creating 120 jars, each labeled with a different spice name, ready to collect samples

---

#### **Line 44:** (Empty line)

---

#### **Line 45:** `# Register hooks to capture activations`

**What it does:**
- Comment explaining the hook installation section

---

#### **Line 46:** `def make_hook(layer_name):`

**What it does:**
- Defines a function that creates hook functions
- This is a "factory function" - a function that makes other functions

**Breaking it down:**
- `def` = Define function
- `make_hook` = Function name
- `(layer_name)` = Input parameter: which layer this hook is for

**Why a factory function?**
- We need a separate hook for each layer (120 hooks)
- Each hook needs to know its own layer name
- `make_hook` creates custom hooks for each layer

**Analogy:** Like a stamp maker: you give it text, it creates a custom stamp with that text

---

#### **Line 47:** `def hook(module, inp, out):`

**What it does:**
- Defines the actual hook function (nested inside `make_hook`)
- This function will be called automatically when data passes through the layer

**Breaking it down:**
- `def hook` = Define function named "hook"
- `(module, inp, out)` = Three parameters PyTorch automatically provides:
  - `module`: The layer itself
  - `inp`: Input to the layer (tuple)
  - `out`: Output from the layer

**What is a hook?**
- A callback function
- Automatically called by PyTorch during model execution
- Like a sensor that activates when something passes by

**Function signature required by PyTorch:**
- Must accept exactly these 3 parameters
- PyTorch calls it automatically: `hook(module, input, output)`

**Analogy:** Like an automatic door sensor: when someone walks through (data passes through layer), it triggers (hook function runs)

---

#### **Line 48:** `if inp and isinstance(inp[0], torch.Tensor):`

**What it does:**
- Safety check: make sure input exists and is valid
- Only process if input is present and is a tensor

**Breaking it down:**

**Part 1:** `if inp`
- Check if input exists (not None or empty)
- `inp` is a tuple, could be empty: `()`

**Part 2:** `and isinstance(inp[0], torch.Tensor)`
- `inp[0]` = First element of input tuple
- `isinstance(inp[0], torch.Tensor)` = Check if it's a tensor
- Returns True if it's a tensor

**Why these checks?**
- Some modules might receive empty input
- Some might receive non-tensor input
- We only want to process actual tensor data

**Example:**
```python
# Good:
inp = (torch.tensor([[1, 2, 3]]),)  → True ✓

# Bad:
inp = ()  → False ✗
inp = None  → False ✗
inp = ("not a tensor",)  → False ✗
```

---

#### **Line 49:** `act = inp[0].detach().reshape(-1, inp[0].size(-1)).abs().mean(dim=0).cpu()`

**What it does:**
- Processes the input to calculate activation strength
- This is the core of the Wanda technique!

**This line does A LOT. Let's break it down step by step:**

**Step 1:** `inp[0]`
- Get first element from input tuple
- Example shape: `[batch_size, sequence_length, hidden_size]` = `[1, 50, 3072]`

**Step 2:** `.detach()`
- Detaches tensor from computation graph
- Prevents gradient tracking (saves memory)
- We're observing, not training
- **Analogy:** Like taking a photo (copy) instead of the original painting

**Step 3:** `.reshape(-1, inp[0].size(-1))`
- Reshapes the tensor into 2D matrix
- `-1` = "figure out this dimension automatically"
- `inp[0].size(-1)` = Last dimension size (hidden_size = 3072)
- **Result:** `[batch_size * sequence_length, hidden_size]` = `[50, 3072]`
- **Why:** Flatten batch and sequence into one dimension
- **Analogy:** Like arranging 3D boxes into a flat grid

**Step 4:** `.abs()`
- Takes absolute value of every number
- Negative values become positive
- **Why:** We care about magnitude, not sign
- **Example:** `[-2, 3, -1]` → `[2, 3, 1]`

**Step 5:** `.mean(dim=0)`
- Calculates average along dimension 0 (rows)
- Results in one value per column
- **Result:** `[hidden_size]` = `[3072]`
- **What this means:** Average activation per input dimension
- **Analogy:** Like calculating average temperature for each month of the year

**Step 6:** `.cpu()`
- Moves tensor from GPU to CPU
- **Why:** Store activations in CPU memory (more RAM available)
- GPU memory is precious, used for model weights

**Final result:**
- 1D tensor with 3072 values
- Each value = average activation for one input dimension
- Higher values = that input dimension activated strongly

**Complete example:**
```python
# Input:
[[0.5, -1.2, 0.8],
 [1.0, -0.5, 0.3],
 [0.2,  0.9, -1.1]]

# After abs():
[[0.5, 1.2, 0.8],
 [1.0, 0.5, 0.3],
 [0.2, 0.9, 1.1]]

# After mean(dim=0):
[0.57, 0.87, 0.73]  ← Average of each column
```

**Analogy:** Like measuring how much each ingredient is used across all dishes, ignoring whether you added or removed it

---

#### **Line 50:** `activations[layer_name].append(act)`

**What it does:**
- Stores the calculated activation in our activations dictionary
- Adds it to the list for this specific layer

**Breaking it down:**
- `activations[layer_name]` = The list for this layer
- `.append(act)` = Add the activation tensor to the list

**Example after processing 3 samples:**
```python
activations["model.layers.0.self_attn.q_proj"] = [
    tensor([0.57, 0.87, 0.73, ...]),  # Sample 1
    tensor([0.62, 0.91, 0.69, ...]),  # Sample 2
    tensor([0.59, 0.85, 0.77, ...]),  # Sample 3
]
```

**Analogy:** Like keeping a log: every time a sensor triggers, write down the measurement in a notebook

---

#### **Line 51:** `return hook`

**What it does:**
- Returns the hook function we just created
- Remember: we're still inside `make_hook`, which creates hooks

**Why return it?**
- The caller needs the hook function to install it
- `make_hook("layer1")` returns a hook configured for "layer1"

**Indentation:**
- Aligned with `def hook`, so it's part of `make_hook`
- NOT part of `hook` function itself

---

#### **Line 52:** (Empty line)

---

#### **Line 53:** `hooks = [module.register_forward_hook(make_hook(name)) for name, module in target_modules.items()]`

**What it does:**
- Creates and installs hooks on all target modules
- Stores hook handles in a list (so we can remove them later)

**Breaking it down:**

This is a list comprehension that does a lot! Let's unpack it:

**Part 1:** `for name, module in target_modules.items()`
- Loop through all target modules
- `name` = layer name (string)
- `module` = layer object

**Part 2:** `make_hook(name)`
- Call our factory function
- Creates a hook configured for this specific layer
- Returns a hook function

**Part 3:** `module.register_forward_hook(...)`
- Installs the hook on this module
- PyTorch will call this hook every time data passes through
- Returns a "hook handle" (reference to remove it later)

**Part 4:** `[... for ...]`
- List comprehension: collect all hook handles in a list

**Result:**
```python
hooks = [
    <hook_handle_1>,
    <hook_handle_2>,
    # ... 120 more handles ...
]
```

**Flow for one module:**
1. `make_hook("model.layers.0.self_attn.q_proj")` creates a hook
2. `module.register_forward_hook(hook)` installs it
3. Returns a handle
4. Handle stored in list

**Why store handles?**
- We need them to remove hooks later
- Like keeping receipts so you can return items

**Analogy:** Like installing 120 security cameras in a building and keeping the list of camera IDs so you can turn them off later

---

#### **Line 54:** (Empty line)

---

#### **Line 55:** `# Run calibration data`

**What it does:**
- Comment explaining next section

---

#### **Line 56:** `print("Collecting activations...")`

**What it does:**
- Informs user that calibration is starting

**Output:**
```
Collecting activations...
```

---

#### **Line 57:** `for text in tqdm(texts, desc="Processing"):`

**What it does:**
- Loop through all 80 Java code samples
- Process each one through the model
- Show a progress bar

**Breaking it down:**
- `for text in` = Loop through
- `tqdm(texts, desc="Processing")` = Wrap list with progress bar
  - `texts` = Our 80 Java code samples
  - `desc="Processing"` = Label for progress bar

**Progress bar output:**
```
Processing: 100%|████████| 80/80 [02:15<00:00,  1.69s/it]
```

**What each part means:**
- `100%` = Completion percentage
- `█████` = Visual bar
- `80/80` = Current/total items
- `[02:15<00:00]` = Time elapsed < Time remaining
- `1.69s/it` = Seconds per item

**Analogy:** Like a loading bar when downloading 80 files

---

#### **Line 58:** `inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)`

**What it does:**
- Converts one Java code sample to numbers
- Prepares it for the model
- Moves it to the same device as the model (GPU/CPU)

**Breaking it down:**

**Part 1:** `tokenizer(text, ...)`
- Call the tokenizer on the text
- `text` = one Java code string

**Parameters:**

**a) `return_tensors="pt"`**
- Return PyTorch tensors (not lists)
- "pt" = PyTorch
- Alternative: "tf" for TensorFlow

**b) `truncation=True`**
- If code is too long, cut it off
- Prevents memory issues

**c) `max_length=256`**
- Maximum 256 tokens
- Longer code will be truncated
- Shorter code will be padded

**Part 2:** `.to(model.device)`**
- Move tensors to same device as model
- If model on GPU → move tensors to GPU
- If model on CPU → keep tensors on CPU
- **Why:** PyTorch requires tensors and model on same device

**Result:**
```python
inputs = {
    'input_ids': tensor([[1898, 7778, 374, ...]]),  # Token IDs
    'attention_mask': tensor([[1, 1, 1, ...]])       # Which tokens to attend to
}
```

**Example:**
```python
text = "public boolean isEven(int n) { return n % 2 == 0; }"

After tokenization:
input_ids = [1898, 7778, 374, 23830, 1577, 308, 8, 314, 471, 308, 1034, 220, 17, 624, 220, 15, 26, 457]
```

**Analogy:** Like translating a sentence from English to numbers and putting it on the same table (device) as your calculator

---

#### **Line 59:** `with torch.no_grad():`

**What it does:**
- Starts a "no gradient" context
- Disables gradient calculation (used for training)

**Breaking it down:**
- `with` = Context manager (runs code in special mode)
- `torch.no_grad()` = PyTorch function that disables gradients

**What are gradients?**
- Mathematical derivatives
- Used during training to update weights
- We're not training, just observing
- Computing them wastes memory and time

**Memory savings:**
- With gradients: 12GB GPU memory
- Without gradients: 6GB GPU memory

**Speed improvement:**
- ~2x faster inference

**Context manager:**
```python
with torch.no_grad():
    # Code here runs without gradients
    ...
# Outside: gradients re-enabled (if they were before)
```

**Analogy:** Like putting your phone in airplane mode to save battery when you don't need connectivity

---

#### **Line 60:** `model(**inputs)`

**What it does:**
- Runs the input through the model
- Triggers all our hooks to record activations
- We ignore the output (just observing activations)

**Breaking it down:**
- `model` = Our loaded language model
- `(**inputs)` = "Unpack dictionary as keyword arguments"

**What is `**` (double star)?**
- Unpacks a dictionary
- `func(**{'a': 1, 'b': 2})` becomes `func(a=1, b=2)`

**So `model(**inputs)` becomes:**
```python
model(input_ids=tensor([[1898, 7778, ...]]),
      attention_mask=tensor([[1, 1, ...]]))
```

**What happens during this call:**
1. Input goes through first layer
2. Hook on first layer triggers → records activation
3. Output goes to second layer
4. Hook on second layer triggers → records activation
5. ... continues through all 120 layers ...
6. Final output generated (we ignore it)

**Why ignore output?**
- We only care about activations, not the generated text
- The output would be random/incomplete anyway (model not generating full response)

**Analogy:** Like running water through pipes to measure flow rates at each junction, but not caring what comes out at the end

---

#### **Line 61:** (Empty line)

---

#### **Line 62:** `# Remove hooks`

**What it does:**
- Comment explaining next section

---

#### **Line 63:** `for h in hooks:`

**What it does:**
- Loop through all hook handles

**Breaking it down:**
- `for h in` = Loop variable
- `hooks` = List of 120 hook handles we saved earlier

---

#### **Line 64:** `h.remove()`

**What it does:**
- Removes one hook from the model
- Uninstalls the sensor

**Why remove hooks?**
- Hooks slow down model execution
- We're done collecting data
- Clean up to restore normal model operation

**What happens:**
- PyTorch stops calling this hook function
- Layer operates normally again

**Analogy:** Like removing all the security cameras after reviewing the footage

---

#### **Line 65:** (Empty line)

---

#### **Line 66:** `# Average activations across all samples`

**What it does:**
- Comment explaining next section

---

#### **Line 67:** `print("Computing activation scales...")`

**What it does:**
- Informs user we're calculating averages

**Output:**
```
Computing activation scales...
```

---

#### **Line 68:** `param_act_scales = {}`

**What it does:**
- Creates empty dictionary for final activation scales
- Will map parameter names to average activations

**What it will store:**
```python
param_act_scales = {
    "model.layers.0.self_attn.q_proj.weight": tensor([0.58, 0.86, ...]),
    # ... 120 more entries ...
}
```

---

#### **Line 69:** `for pname, mname in param_to_module.items():`

**What it does:**
- Loop through our parameter→module mapping

**Breaking it down:**
- `pname` = parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
- `mname` = module name (e.g., "model.layers.0.self_attn.q_proj")
- `param_to_module.items()` = All key-value pairs in dictionary

---

#### **Line 70:** `if activations[mname]:`

**What it does:**
- Check if we collected any activations for this module
- Only process if list is not empty

**Why check?**
- Safety check
- In theory, all modules should have activations
- In practice, some edge cases might have empty lists

---

#### **Line 71:** `param_act_scales[pname] = torch.stack(activations[mname]).mean(dim=0)`

**What it does:**
- Calculates average activation across all 80 samples
- Stores result in our final dictionary

**Breaking it down:**

**Part 1:** `activations[mname]`
- Get list of activations for this module
- List of 80 tensors: `[tensor1, tensor2, ..., tensor80]`
- Each tensor shape: `[3072]`

**Part 2:** `torch.stack(...)`
- Stack all tensors into one big tensor
- Changes list of tensors into single tensor
- **Before:** 80 separate tensors of shape `[3072]`
- **After:** One tensor of shape `[80, 3072]`
- **Analogy:** Like stacking 80 sheets of paper into one pile

**Part 3:** `.mean(dim=0)`
- Calculate average along dimension 0 (across the 80 samples)
- **Result:** `[3072]` (one value per dimension)
- **What it means:** Average activation for each input dimension across all samples

**Example:**
```python
# We have:
activations = [
    tensor([0.5, 0.9]),  # Sample 1
    tensor([0.7, 0.8]),  # Sample 2
    tensor([0.6, 1.0]),  # Sample 3
]

# Stack:
stacked = tensor([[0.5, 0.9],
                  [0.7, 0.8],
                  [0.6, 1.0]])  # Shape: [3, 2]

# Mean:
result = tensor([0.6, 0.9])  # Average each column
```

**Part 4:** `param_act_scales[pname] = ...`
- Store result in dictionary
- Key: parameter name
- Value: average activation tensor

**Analogy:** Like calculating each student's average test score across 80 tests

---

#### **Line 72:** (Empty line)

---

#### **Line 73:** `return param_act_scales`

**What it does:**
- Returns the final dictionary of activation scales
- Ends the `collect_activations` function

**What's returned:**
```python
{
    "model.layers.0.self_attn.q_proj.weight": tensor([0.58, 0.86, 0.72, ...]),  # 3072 values
    "model.layers.0.self_attn.k_proj.weight": tensor([0.61, 0.83, 0.69, ...]),
    # ... 120 more entries ...
}
```

**This data is crucial:**
- Tells us which parameters activate strongly
- Used in Wanda formula: Importance = |Weight| × Activation
- Without this, we'd only prune by weight size (less effective)

---

#### **Line 74:** (Empty line)

---

### SECTION 6: Wanda Pruning Function (Lines 75-104)

#### **Line 75:** `# Wanda Pruning`

**What it does:**
- Comment marking the pruning function section

---

#### **Line 76:** `def wanda_prune(model, sparsity, param_act_scales):`

**What it does:**
- Defines the function that performs actual pruning

**Breaking it down:**
- `def wanda_prune` = Define function named "wanda_prune"
- Parameters:
  - `model`: The LLM to prune
  - `sparsity`: How much to prune (0.05 = 5%)
  - `param_act_scales`: Activation data we collected

---

#### **Line 77:** `"""Prunes model using Wanda (Weights AND Activations) technique."""`

**What it does:**
- Docstring explaining the function

---

#### **Line 78:** `pruned_count = total_params = 0`

**What it does:**
- Initializes two counters to zero
- Will track statistics during pruning

**Breaking it down:**
- `pruned_count` = How many parameters we set to zero
- `total_params` = Total parameters we processed
- `= 0` = Start both at zero
- Chained assignment: both variables set to 0

**Why track these?**
- Calculate actual sparsity at the end
- Verify pruning worked correctly
- Report to user

**Analogy:** Like keeping a tally: "Items removed" and "Items checked"

---

#### **Line 79:** (Empty line)

---

#### **Line 80:** `for name, param in model.named_parameters():`

**What it does:**
- Loop through every parameter in the entire model
- Check each one to see if we should prune it

**Breaking it down:**
- `for name, param in` = Loop through pairs
- `model.named_parameters()` = Returns all parameters with names
  - `name`: parameter name (string)
  - `param`: parameter tensor (the actual numbers)

**How many parameters?**
- ~3 billion parameters in our model
- But they're organized in ~300 parameter tensors
- So this loop runs ~300 times, not 3 billion times

**Example pairs:**
```python
("model.layers.0.self_attn.q_proj.weight", tensor([[...]]))
("model.layers.0.self_attn.q_proj.bias", tensor([...]))
("model.layers.0.self_attn.k_proj.weight", tensor([[...]]))
...
```

---

#### **Line 81:** `if name in param_act_scales and param.dim() == 2:`

**What it does:**
- Checks if we should prune this parameter
- Two conditions must be True

**Breaking it down:**

**Condition 1:** `name in param_act_scales`
- Check if we have activation data for this parameter
- Only True for our 120 target parameters
- False for all others (embeddings, biases, normalizations)

**Condition 2:** `param.dim() == 2`
- Check if parameter is 2D (a matrix)
- `.dim()` returns number of dimensions
- `== 2` checks if it's exactly 2D

**Why 2D only?**
- Linear layer weights are 2D matrices
- Biases are 1D vectors (we don't prune these)
- Embeddings might be 2D but aren't in our activation data

**Example checks:**
```python
# Weight matrix - Process!
param.shape = [3072, 3072]  → dim() = 2  →  True ✓

# Bias vector - Skip!
param.shape = [3072]  → dim() = 1  →  False ✗

# Not in activation data - Skip!
name not in param_act_scales  →  False ✗
```

**Result:**
- Only ~120 parameters pass this check
- The most important ones (attention & MLP weights)

---

#### **Line 82:** `act_scale = param_act_scales[name].to(param.device)`

**What it does:**
- Gets the activation scale for this parameter
- Moves it to the same device as the parameter

**Breaking it down:**
- `param_act_scales[name]` = Look up activation scale
- `.to(param.device)` = Move to same device as param

**Why move to same device?**
- `act_scale` might be on CPU (where we stored it)
- `param` is on GPU (where model lives)
- PyTorch requires both on same device for multiplication

**Example:**
```python
# Get activation scale
act_scale = tensor([0.58, 0.86, 0.72, ...])  # Shape: [3072]

# If param is on GPU, move act_scale to GPU
act_scale = act_scale.to('cuda')
```

---

#### **Line 83:** (Empty line)

---

#### **Line 84:** `# Verify dimensions match`

**What it does:**
- Comment explaining next section

---

#### **Line 85:** `if act_scale.size(0) != param.size(1):`

**What it does:**
- Safety check: verify activation and parameter dimensions match
- If they don't match, skip this parameter

**Breaking it down:**
- `act_scale.size(0)` = Number of elements in activation scale
- `param.size(1)` = Number of columns in parameter matrix (input dimension)
- `!=` = Not equal to

**Why this check?**
- Activation scale length should equal parameter input dimension
- If mismatched, multiplication won't work
- Could indicate data collection error

**Example:**
```python
# Correct:
param.shape = [4096, 3072]  # Output×Input
act_scale.shape = [3072]    # Input dimension
→ Match! ✓

# Incorrect (skip):
param.shape = [4096, 3072]
act_scale.shape = [4096]    # Wrong!
→ Mismatch ✗
```

---

#### **Line 86:** `continue`

**What it does:**
- Skip to next iteration of the loop
- Don't process this parameter

**When executed:**
- Only if dimensions don't match (line 85)

**What `continue` does:**
- Immediately jump to next loop iteration
- Skip all remaining code in this iteration
- Like "next!" in a queue

**Analogy:** Like skipping a damaged item on an assembly line

---

#### **Line 87:** (Empty line)

---

#### **Line 88:** `# Calculate importance: |Weight| × Activation`

**What it does:**
- Comment explaining the core Wanda formula

---

#### **Line 89:** `importance = param.abs() * act_scale.unsqueeze(0)`

**What it does:**
- Calculates importance for each weight
- This is the core of Wanda!

**Breaking it down:**

**Part 1:** `param.abs()`
- Take absolute value of all weights
- Negative weights become positive
- Example: `[2.5, -1.3, 0.8]` → `[2.5, 1.3, 0.8]`

**Part 2:** `act_scale.unsqueeze(0)`
- Add a dimension at position 0
- Changes from 1D to 2D
- **Before:** `[3072]`
- **After:** `[1, 3072]`
- **Why:** So it can broadcast (multiply) with 2D param

**What is broadcasting?**
- NumPy/PyTorch feature for arrays of different shapes
- Automatically repeats smaller array to match larger one
- Example:
  ```python
  [1, 3072] × [4096, 3072]
  →
  [[a, b, c],      [[w1, w2, w3],
   [a, b, c],   ×   [x1, x2, x3],
   ...   ]          ...        ]
  ```

**Part 3:** `param.abs() * act_scale.unsqueeze(0)`
- Element-wise multiplication
- Each weight multiplied by its corresponding activation

**Complete example:**
```python
# Parameter weights (simplified):
param = [[ 2.0, -1.5,  0.3],
         [ 0.1,  3.2, -0.8]]  # Shape: [2, 3]

# Absolute value:
param.abs() = [[2.0, 1.5, 0.3],
               [0.1, 3.2, 0.8]]

# Activation scale:
act_scale = [0.5, 0.9, 0.1]  # Shape: [3]

# Unsqueeze:
act_scale.unsqueeze(0) = [[0.5, 0.9, 0.1]]  # Shape: [1, 3]

# Multiply (broadcast):
importance = [[2.0*0.5, 1.5*0.9, 0.3*0.1],
              [0.1*0.5, 3.2*0.9, 0.8*0.1]]
           = [[1.0, 1.35, 0.03],
              [0.05, 2.88, 0.08]]
```

**Result:**
- 2D tensor same shape as parameter
- Each value = importance of that specific weight
- Higher = more important = keep
- Lower = less important = prune

**This is Wanda's innovation:**
- Not just weight size (magnitude pruning)
- Not just activation (activation pruning)
- BOTH combined for smarter decisions!

---

#### **Line 90:** (Empty line)

---

#### **Line 91:** `# Find threshold (keep top (1-sparsity)% most important)`

**What it does:**
- Comment explaining threshold calculation

---

#### **Line 92:** `k = int((1 - sparsity) * importance.numel())`

**What it does:**
- Calculates how many parameters to KEEP
- The rest will be pruned

**Breaking it down:**

**Part 1:** `importance.numel()`
- `.numel()` = Number of elements in tensor
- Counts total weights in this parameter
- Example: `[4096, 3072]` → `4096 × 3072 = 12,582,912`

**Part 2:** `(1 - sparsity)`
- `sparsity = 0.05` (5% pruning)
- `1 - 0.05 = 0.95` (95% keeping)

**Part 3:** `(1 - sparsity) * importance.numel()`
- Multiply: keep 95% of total
- Example: `0.95 × 12,582,912 = 11,953,766`

**Part 4:** `int(...)`
- Convert to integer (round down)
- Can't keep fractional weights!

**Result:**
```python
sparsity = 0.05
numel = 12,582,912
k = int(0.95 × 12,582,912)
k = 11,953,766

Meaning: Keep top 11,953,766 weights, prune the other 629,146
```

**Analogy:** Like deciding to keep the top 95% of students based on test scores

---

#### **Line 93:** `threshold = torch.topk(importance.flatten(), k, largest=True).values.min() if k > 0 else importance.max()`

**What it does:**
- Finds the threshold value
- Weights above threshold = keep
- Weights below threshold = prune

---

### 🚨 **CRITICAL BUG FIX ALERT** 🚨

**This line contained a major bug in earlier versions that completely broke the pruning!**

#### The Bug:

**❌ WRONG CODE (caused model to crash):**
```python
k = int((1 - sparsity) * flat.numel())
if k <= 0:  # ← BUG: Condition is backwards!
    threshold = torch.topk(flat, k, largest=True).values.min()
else:
    threshold = flat.max()  # ← This runs when k > 0!
```

**✅ CORRECT CODE (current version):**
```python
k = int((1 - sparsity) * importance.numel())
threshold = torch.topk(importance.flatten(), k, largest=True).values.min() if k > 0 else importance.max()
```

#### What Went Wrong:

**The Math:**
- With 5% sparsity: `k = 0.95 × 12,582,912 = 11,953,766`
- `k` is a large positive number (not <= 0)
- So the buggy code went to the `else` branch: `threshold = flat.max()`

**The Effect:**
```python
# Buggy version:
k = 11,953,766  # Want to keep 95% of weights
if k <= 0:      # False! (11M is not <= 0)
    # This branch never executes
else:
    threshold = flat.max()  # Sets threshold to MAXIMUM value!

# Then:
mask = importance > threshold  # Keep only weights GREATER than max
# Problem: Almost nothing is greater than the maximum!
# Result: Pruned 95-99% instead of 5%!
```

**Example with real numbers:**
```python
importance = [2.88, 1.35, 1.0, 0.08, 0.05, 0.03]

# Buggy version:
threshold = max(importance) = 2.88
mask = importance > 2.88
mask = [False, False, False, False, False, False]
# EVERYTHING PRUNED! 100% sparsity!

# Correct version:
k = 5  # Keep top 5
threshold = min(top 5 values) = 0.05
mask = importance >= 0.05
mask = [True, True, True, True, True, False]
# Only 1 pruned, 5 kept - correct!
```

#### Impact on Model Output:

**With the bug (95%+ parameters zeroed):**
```
Input:  "Hello"
Output: "HelloHelloHelloHelloHello..."

Input:  "Write a Java function"
Output: "Write a Java functionfunctionfunctionfunction..."

Input:  "Explain recursion"
Output: "Explain recursionrecursionrecursionrecursion..."
```

**Why repetition?**
- Attention mechanism destroyed (can't track what was already generated)
- Context understanding broken
- Model falls into repetitive loops
- Like someone with severe short-term memory loss repeating themselves

**After fix (5% parameters zeroed):**
```
Input:  "Write a Java function that returns true if a number is even."
Output: "public boolean isEven(int number) {
            return number % 2 == 0;
        }"
```

#### Technical Explanation:

**What got destroyed:**
1. **Attention layers:** Can't determine which previous words to focus on
2. **Context tracking:** Model forgets what it already generated
3. **Output diversity:** Limited options force repetition
4. **Coherence:** No logical flow between tokens

**Why 5% pruning works but 95% doesn't:**
- **5% pruning:** Removes least important connections, model adapts
- **95% pruning:** Removes critical pathways, model cannot function
- It's like removing 5% of roads vs. removing 95% of roads in a city

**Analogy:**
- **5% pruning:** Trimming dead branches from a tree - tree stays healthy
- **95% pruning:** Cutting down the entire tree except a few leaves - tree dies

---

**Breaking it down (complex line!):**

**Part 1:** `importance.flatten()`
- Flattens 2D tensor to 1D
- **Before:** `[[1.0, 1.35], [0.05, 2.88]]`
- **After:** `[1.0, 1.35, 0.05, 2.88]`
- **Why:** `topk` needs 1D input

**Part 2:** `torch.topk(..., k, largest=True)`
- Finds the k largest values
- `k` = number to keep (e.g., 11,953,766)
- `largest=True` = get biggest values (not smallest)
- Returns: object with `.values` and `.indices`

**Part 3:** `.values`
- Extracts just the values (not indices)
- Returns sorted tensor of top k values
- Example: `[2.88, 1.35, 1.0, ...]` (largest first)

**Part 4:** `.min()`
- Gets minimum of the top k values
- This is our threshold!
- Example: If top k are `[2.88, 1.35, 1.0, 0.08, 0.05]`, min is `0.05`

**Part 5:** `if k > 0 else importance.max()`
- Ternary conditional
- If `k > 0`: do everything above
- Else (if `k == 0`): set threshold to max (keeps nothing)
- Safety check for edge case

**Complete example:**
```python
importance = [2.88, 1.35, 1.0, 0.08, 0.05, 0.03]
k = 5  # Keep top 5

# Step 1: topk
topk_result = [2.88, 1.35, 1.0, 0.08, 0.05]

# Step 2: min
threshold = 0.05

# Result: Keep everything >= 0.05
```

**Why this works:**
- Top k values are all important
- Minimum of top k is the "cutoff"
- Everything >= cutoff is in top k → keep
- Everything < cutoff is not in top k → prune

**Analogy:** Like grading on a curve: find the k best scores, see what the lowest of those is, that's the passing grade

---

#### **Line 94:** (Empty line)

---

#### **Line 95:** `# Apply mask (keep important, zero out unimportant)`

**What it does:**
- Comment explaining masking section

---

#### **Line 96:** `mask = importance >= threshold`

**What it does:**
- Creates a binary mask (True/False or 1/0)
- True where weight is important, False where unimportant

**Breaking it down:**
- `importance >= threshold` = Element-wise comparison
- `>=` = Greater than or equal to
- Returns boolean tensor (True/False)

**Example:**
```python
importance = [[1.0, 1.35, 0.03],
              [0.05, 2.88, 0.08]]

threshold = 0.05

mask = [[True,  True,  False],
        [True,  True,  True ]]
```

**In PyTorch:**
- True converts to 1
- False converts to 0
- So mask is effectively `[[1, 1, 0], [1, 1, 1]]`

**Analogy:** Like marking items to keep (✓) vs. items to throw away (✗)

---

#### **Line 97:** `param.data *= mask`

**What it does:**
- Applies the mask to the actual parameter weights
- Zeros out unimportant weights
- **This is where pruning actually happens!**

**Breaking it down:**
- `param.data` = The actual weight tensor (not a wrapper)
- `*=` = In-place multiplication
- `mask` = Binary mask (1s and 0s)

**What happens:**
```python
# Before:
param.data = [[ 2.0, -1.5,  0.3],
              [ 0.1,  3.2, -0.8]]

# Mask:
mask = [[1, 1, 0],
        [1, 1, 1]]

# After multiplication:
param.data = [[ 2.0, -1.5,  0.0],  ← 0.3 became 0.0
              [ 0.1,  3.2, -0.8]]

Pruned: 1 weight out of 6
```

**Why `param.data`?**
- Direct access to tensor values
- Modifies in-place (saves memory)
- Doesn't track gradients (we're not training)

**What `*=` does:**
- In-place operation (modifies original)
- Equivalent to: `param.data = param.data * mask`
- More memory efficient

**This is permanent:**
- Those weights are now zero
- Model will no longer use them
- They contribute nothing to computations

**Analogy:** Like erasing marks from a whiteboard - they're gone permanently

---

#### **Line 98:** (Empty line)

---

#### **Line 99:** `# Track statistics`

**What it does:**
- Comment explaining stat tracking

---

#### **Line 100:** `pruned_count += (~mask).sum().item()`

**What it does:**
- Counts how many weights were set to zero
- Adds to running total

**Breaking it down:**

**Part 1:** `~mask`
- `~` = Bitwise NOT operator
- Flips True→False and False→True
- Inverts the mask
- Example: `[True, True, False]` → `[False, False, True]`

**Why invert?**
- Mask has True where we KEPT weights
- We want to count where we PRUNED weights
- Inverted mask has True where we pruned

**Part 2:** `.sum()`
- Counts True values
- True = 1, False = 0
- Sum gives count of True
- Example: `[False, False, True]`.sum() = 1

**Part 3:** `.item()`
- Converts single-value tensor to Python number
- tensor(1) → 1 (integer)
- **Why:** For accumulation in Python variable

**Part 4:** `pruned_count +=`
- Add to running total
- `+=` = increment by
- Keeps track across all layers

**Example:**
```python
# Layer 1:
mask = [[1, 1, 0], [1, 1, 1]]
~mask = [[0, 0, 1], [0, 0, 0]]
(~mask).sum() = 1
pruned_count = 0 + 1 = 1

# Layer 2:
mask = [[1, 0], [0, 1]]
~mask = [[0, 1], [1, 0]]
(~mask).sum() = 2
pruned_count = 1 + 2 = 3

# Total pruned: 3
```

---

#### **Line 101:** `total_params += mask.numel()`

**What it does:**
- Counts total parameters in this layer
- Adds to running total

**Breaking it down:**
- `mask.numel()` = Number of elements in mask
- Same as number of weights in this parameter
- `total_params +=` = Add to running total

**Example:**
```python
# Layer 1: [4096, 3072]
total_params = 0 + (4096 × 3072) = 12,582,912

# Layer 2: [3072, 3072]
total_params = 12,582,912 + (3072 × 3072) = 22,019,296

# Etc...
```

**Why track this?**
- Calculate actual sparsity percentage
- Verify we processed expected number of parameters
- Report to user

---

#### **Line 102:** (Empty line)

---

#### **Line 103:** `print(f"Pruned {pruned_count:,} / {total_params:,} params ({pruned_count/total_params:.2%})")`

**What it does:**
- Prints final statistics
- Shows how many parameters were pruned

**Breaking it down:**

**Part 1:** `print(f"...")`
- f-string for formatting

**Part 2:** `{pruned_count:,}`
- Insert `pruned_count` value
- `:,` = Add commas as thousands separator
- Example: `150245823` → `150,245,823`

**Part 3:** `{total_params:,}`
- Same formatting for total

**Part 4:** `{pruned_count/total_params:.2%}`
- Divide pruned by total
- `:.2%` = Format as percentage with 2 decimal places
- Example: `0.05` → `5.00%`

**Output:**
```
Pruned 150,245,823 / 3,004,916,460 params (5.00%)
```

**What this tells us:**
- Out of ~3 billion parameters
- We pruned ~150 million (set to zero)
- That's exactly 5%
- Target achieved!

---

#### **Line 104:** `return model`

**What it does:**
- Returns the pruned model
- Ends the function

**What's returned:**
- Same model object, but with 5% of weights set to zero
- Model is modified in-place, but we return it anyway (convention)

---

#### **Line 105:** (Empty line)

---

### SECTION 7: Execute Pruning (Lines 106-110)

#### **Line 106:** `# Execute Pruning`

**What it does:**
- Comment marking execution section

---

#### **Line 107:** `print("\n" + "="*50)`

**What it does:**
- Prints a visual separator line
- Makes output easier to read

**Breaking it down:**
- `"\n"` = Newline character (blank line)
- `"="*50` = Repeat "=" 50 times
- `+` = Concatenate strings

**Output:**
```

==================================================
```

**Why:**
- Visual organization
- Separates sections of output
- Makes terminal output cleaner

**Analogy:** Like drawing a horizontal line in notes to separate topics

---

#### **Line 108:** `param_act_scales = collect_activations(model, tokenizer, texts)`

**What it does:**
- Calls our activation collection function
- Runs all 80 Java samples through the model
- Returns activation data

**Breaking it down:**
- `param_act_scales` = Variable to store results
- `collect_activations(...)` = Our function from earlier
- Passing 3 arguments:
  - `model`: The loaded LLM
  - `tokenizer`: Text↔number converter
  - `texts`: 80 Java code samples

**What happens (takes 2-5 minutes):**
1. Install hooks on 120 layers
2. Process each of 80 samples:
   - Tokenize code
   - Run through model
   - Hooks record activations
3. Remove hooks
4. Calculate average activations
5. Return dictionary

**Progress shown:**
```
Targeting 120 layers for pruning
Collecting activations...
Processing: 100%|████████| 80/80 [02:15<00:00,  1.69s/it]
Computing activation scales...
```

**Result:**
```python
param_act_scales = {
    "model.layers.0.self_attn.q_proj.weight": tensor([...]),
    # ... 120 entries ...
}
```

---

#### **Line 109:** `print("="*50 + "\n")`

**What it does:**
- Prints another separator line
- Sandwiches the activation collection output

**Output:**
```
==================================================

```

---

#### **Line 110:** `model = wanda_prune(model, SPARSITY, param_act_scales)`

**What it does:**
- Calls our pruning function
- Actually performs the pruning
- Returns pruned model

**Breaking it down:**
- `model =` = Store result (overwrites original)
- `wanda_prune(...)` = Our pruning function
- Arguments:
  - `model`: The LLM
  - `SPARSITY`: 0.05 (5%)
  - `param_act_scales`: Activation data we just collected

**What happens (takes 1-2 minutes):**
1. Loop through all parameters
2. For each target parameter:
   - Calculate importance (|Weight| × Activation)
   - Find threshold
   - Create mask
   - Zero out unimportant weights
3. Print statistics

**Output:**
```
Pruned 150,245,823 / 3,004,916,460 params (5.00%)
```

**Result:**
- Same model object
- But 5% of weights now zero
- Model still functional (hopefully!)

---

#### **Line 111:** (Empty line)

---

### SECTION 8: Save Pruned Model (Lines 112-116)

#### **Line 112:** `# Save Pruned Model`

**What it does:**
- Comment marking save section

---

#### **Line 113:** `print(f"\nSaving to {SAVE_DIR}...")`

**What it does:**
- Informs user where model is being saved

**Output:**
```
Saving to ./QwenCoder3B_JavaPruned-5...
```

---

#### **Line 114:** `model.save_pretrained(SAVE_DIR, safe_serialization=True)`

**What it does:**
- Saves the pruned model weights to disk
- Creates directory and saves all necessary files

**Breaking it down:**
- `model.save_pretrained(...)` = Hugging Face function
- `SAVE_DIR` = "./QwenCoder3B_JavaPruned-5"
- `safe_serialization=True` = Use SafeTensors format

**What gets saved:**
- `model.safetensors`: Pruned weights (~6GB)
- `config.json`: Model configuration
- `generation_config.json`: Generation settings

**What is SafeTensors?**
- Secure format for saving model weights
- Prevents malicious code injection
- Faster loading than pickle
- Industry standard

**Time taken:**
- 30-60 seconds to write 6GB

**Analogy:** Like saving a massive edited video file to your hard drive

---

#### **Line 115:** `tokenizer.save_pretrained(SAVE_DIR)`

**What it does:**
- Saves the tokenizer to the same directory
- Necessary to use the model later

**What gets saved:**
- `tokenizer.json`: Vocabulary and rules
- `tokenizer_config.json`: Configuration
- `special_tokens_map.json`: Special tokens

**Why save tokenizer?**
- Model needs tokenizer to work
- Must be exact same tokenizer used during training
- Ensures consistent text↔number conversion

**File size:**
- ~2-5 MB (small)

---

#### **Line 116:** `print(f"✅ Model saved successfully!")`

**What it does:**
- Confirmation message
- Lets user know save completed

**Output:**
```
✅ Model saved successfully!
```

**Checkmark emoji:**
- Visual indicator of success
- Makes output friendlier

---

#### **Line 117:** (Empty line)

---

### SECTION 9: Validation Test (Lines 118-127)

#### **Line 118:** `# Validation Test`

**What it does:**
- Comment marking validation section

---

#### **Line 119:** `print("\n" + "="*50)`

**What it does:**
- Visual separator

**Output:**
```

==================================================
```

---

#### **Line 120:** `print("Validation Test:")`

**What it does:**
- Section header

**Output:**
```
Validation Test:
```

---

#### **Line 121:** `print("="*50)`

**What it does:**
- Another separator line

**Output:**
```
==================================================
```

---

#### **Line 122:** `prompt = "Write a Java function that returns true if a number is even."`

**What it does:**
- Sets the test prompt
- Simple coding task to verify model works

**Why this prompt?**
- Tests if model can generate Java code
- Simple enough that any working model should handle it
- If model is broken, will produce gibberish or repetition

**Analogy:** Like asking a calculator to solve 2+2 to verify it's working

---

#### **Line 123:** `inputs = tokenizer(prompt, return_tensors="pt").to(model.device)`

**What it does:**
- Converts prompt to numbers
- Prepares for model input

**Breaking it down:**
- Same as line 58 in calibration
- Tokenizes the prompt
- Returns PyTorch tensors
- Moves to model's device

**Result:**
```python
inputs = {
    'input_ids': tensor([[9842, 264, 8102, 734, ...]),
    'attention_mask': tensor([[1, 1, 1, ...]])
}
```

---

#### **Line 124:** `with torch.no_grad():`

**What it does:**
- Disable gradient tracking
- Saves memory during generation

**Why:**
- Not training, just generating
- Gradients waste memory
- Faster inference

---

#### **Line 125:** `outputs = model.generate(**inputs, max_new_tokens=80)`

**What it does:**
- Generates text from the pruned model
- Creates up to 80 new tokens

**Breaking it down:**
- `model.generate()` = Text generation function
- `**inputs` = Unpack input dictionary
- `max_new_tokens=80` = Generate up to 80 tokens

**What happens:**
1. Model reads the prompt
2. Predicts next token
3. Adds it to sequence
4. Predicts next token
5. Repeats 80 times or until done

**Good output:**
```
outputs = tensor([[9842, 264, 8102, 734, ..., 457]])
(full sequence including prompt + generated text)
```

**Bad output (if broken):**
```
outputs = tensor([[9842, 264, 264, 264, 264, ...]])
(repetitive - model is broken!)
```

---

#### **Line 126:** `print(tokenizer.decode(outputs[0], skip_special_tokens=True))`

**What it does:**
- Converts generated tokens back to text
- Prints it for user to see

**Breaking it down:**
- `outputs[0]` = First (only) sequence
- `tokenizer.decode(...)` = Convert numbers→text
- `skip_special_tokens=True` = Hide special tokens (e.g., <|endoftext|>)
- `print(...)` = Display result

**Good output:**
```
Write a Java function that returns true if a number is even.

public boolean isEven(int number) {
    return number % 2 == 0;
}
```

**Bad output (broken model):**
```
Write a Java function that returns true if a number is even.even.even.even.even.
```

**Why this matters:**
- Quickly verifies pruning worked
- If output is good → success!
- If output is broken → something went wrong

---

#### **Line 127:** `print("="*50)`

**What it does:**
- Final separator line
- Closes the output section

**Output:**
```
==================================================
```

---

## Summary

The code performs these steps:

1. **Import libraries** (lines 1-4)
2. **Set configuration** (lines 6-10): Model name, 5% sparsity, 80 samples
3. **Load model** (lines 12-16): Download and load 3B parameter model
4. **Load data** (lines 18-26): Get 80 Java code samples
5. **Collect activations** (lines 28-73): Measure which parameters activate
6. **Prune model** (lines 75-104): Zero out least important 5% of weights
7. **Execute** (lines 106-110): Run the pruning process
8. **Save** (lines 112-116): Save pruned model to disk
9. **Validate** (lines 118-127): Test that model still works

**Total:** 127 lines of code that reduce a 3B parameter model by 5% while maintaining functionality!

---

## Expected Results

### What Success Looks Like:

#### 1. During Pruning:
```
Loading model...
✓ Model loaded: Qwen2.5-Coder-3B-Instruct (3B parameters)

Loading Java calibration samples...
✓ Using 80 calibration samples

Collecting activations from calibration data...
Processing samples: 100%|██████████| 80/80 [02:15<00:00,  1.69s/it]
✓ Targeting 120 linear layers for pruning
✓ Computing activation scales per layer...

Pruning model with Wanda...
✓ Pruned 150,245,823 / 3,004,916,460 params (5.00%)

✅ Pruned model saved to ./QwenCoder3B_JavaPruned-10

Testing a short generation to validate pruning...
Write a Java function that returns true if a number is even.

public boolean isEven(int number) {
    return number % 2 == 0;
}
```

#### 2. File Size Comparison:
- **Original model:** ~6.0 GB
- **Pruned model:** ~5.7 GB (5% smaller)
- **Note:** Size reduction is modest because we still store the zeros

#### 3. Performance:
- **Speed:** 5-10% faster inference (fewer computations)
- **Quality:** 95-98% of original quality (minimal degradation)
- **Memory:** Slightly lower GPU memory usage

#### 4. Validation:
The model should still be able to:
- ✅ Generate coherent Java code
- ✅ Follow instructions
- ✅ Complete functions logically
- ✅ Use proper syntax

---

## Troubleshooting

### Problem 1: Repetitive Output ("hellohellohello") - THE THRESHOLD BUG

This was a **critical bug** discovered during development that completely broke the model's ability to generate coherent text.

#### **Symptoms:**

The pruned model produces repetitive, looping output:

```
Input:  "Hello"
Output: "HelloHelloHelloHelloHelloHelloHello..."

Input:  "Write a Java function"
Output: "Write a Java functionfunctionfunctionfunction..."

Input:  "public class"
Output: "public classpublicpublicpublicpublicpublic..."

Input:  "Explain recursion"
Output: "Explain recursionrecursionrecursionrecursion..."
```

**Characteristics:**
- Model repeats the last word or phrase over and over
- No coherent sentence structure
- Doesn't follow instructions
- Cannot maintain context
- Loops indefinitely until max tokens reached

---

#### **Root Cause: Inverted Threshold Logic**

**The bug was in Line 93** of the pruning function:

**❌ BUGGY CODE:**
```python
k = int((1 - sparsity) * flat.numel())
if k <= 0:  # ← WRONG! Condition backwards
    threshold = torch.topk(flat, k, largest=True).values.min()
else:
    threshold = flat.max()  # ← This executed for normal case!
mask = importance > threshold
```

**✅ FIXED CODE:**
```python
k = int((1 - sparsity) * importance.numel())
threshold = torch.topk(importance.flatten(), k, largest=True).values.min() if k > 0 else importance.max()
mask = importance >= threshold
```

---

#### **What Went Wrong: Step-by-Step**

**Step 1: Calculate k (number to keep)**
```python
sparsity = 0.05  # Want 5% sparsity
total_params = 12,582,912  # Example layer
k = int((1 - 0.05) * 12,582,912)
k = 11,953,766  # Should keep ~12M parameters
```

**Step 2: Buggy threshold calculation**
```python
if k <= 0:  # Check if k is zero or negative
    # k = 11,953,766, so this is False
    # This branch never executes!
else:
    # Goes here instead!
    threshold = flat.max()  # Sets threshold to MAXIMUM value
    # If max importance = 2.88, threshold = 2.88
```

**Step 3: Create mask**
```python
mask = importance > threshold  # Greater than (not >=)
# Example: importance > 2.88
# Almost nothing is greater than the maximum!
```

**Step 4: Result**
```python
importance = [2.88, 1.35, 1.0, 0.08, 0.05, 0.03]
threshold = 2.88  # The maximum

mask = [False, False, False, False, False, False]
# Everything gets pruned! 100% sparsity instead of 5%!
```

---

#### **Mathematical Impact**

**Intended behavior (5% sparsity):**
```
Total parameters: 3,000,000,000
Should keep:      2,850,000,000 (95%)
Should prune:       150,000,000 (5%)
```

**Actual behavior (buggy code):**
```
Total parameters: 3,000,000,000
Actually kept:       ~30,000,000 (1%)
Actually pruned:  2,970,000,000 (99%)

Result: 99% of the model destroyed!
```

---

#### **Why Repetition Happens**

When 99% of parameters are zeroed:

**1. Attention Mechanism Breaks**
- Can't calculate proper attention scores
- Can't focus on relevant previous tokens
- Loses track of what was already said

**2. Context Window Collapses**
- Model can only "remember" last 1-2 tokens
- Forgets the beginning of its own output
- Like severe short-term memory loss

**3. Output Diversity Destroyed**
- Limited pathways through network
- Only a few token choices remain
- Falls into repetitive loops

**4. Autoregressive Failure**
- Each token depends on previous tokens
- With no context, generates same token
- Loop: see last word → generate last word → repeat

**Example sequence:**
```
Prompt: "Write a Java"
Token 1: "function" (ok)
Token 2: "function" (repeated - no context to vary)
Token 3: "function" (stuck in loop)
Token 4: "function" (continuing loop)
...
```

---

#### **Why the Fix Works**

**Correct logic:**
```python
if k > 0:
    # Normal case: k is positive (e.g., 11,953,766)
    threshold = torch.topk(...).values.min()
    # Threshold = minimum of top 95% = cutoff value
else:
    # Edge case: k is 0 (100% sparsity)
    threshold = importance.max()
    # Prune everything
```

**Result:**
```python
importance = [2.88, 1.35, 1.0, 0.08, 0.05, 0.03]
k = 5  # Keep top 5

# Find top 5:
top_5 = [2.88, 1.35, 1.0, 0.08, 0.05]

# Threshold = minimum of top 5:
threshold = 0.05

# Mask: keep everything >= 0.05
mask = [True, True, True, True, True, False]

# Result: 1 pruned, 5 kept ✓
```

---

#### **How to Verify the Fix**

**Check 1: Inspect the code**
Look at line 93 in `java_llm_prune.py`:
```python
threshold = torch.topk(importance.flatten(), k, largest=True).values.min() if k > 0 else importance.max()
```
✓ Should have `if k > 0` (not `if k <= 0`)

**Check 2: Run the code**
The output should show:
```
Pruned 150,245,823 / 3,004,916,460 params (5.00%)
```
✓ Should be around 5%, not 95%+

**Check 3: Test generation**
After pruning, test with:
```python
prompt = "Write a Java function"
```

✓ **Good output:**
```java
public boolean isEven(int number) {
    return number % 2 == 0;
}
```

✗ **Bad output (bug present):**
```
Write a Java functionfunctionfunctionfunction...
```

---

#### **Comparison: Before vs After**

| Aspect | With Bug | After Fix |
|--------|----------|-----------|
| **Sparsity** | ~99% | ~5% |
| **Params pruned** | 2.97B / 3B | 150M / 3B |
| **Params kept** | 30M | 2.85B |
| **Output quality** | Gibberish/loops | Coherent |
| **Attention** | Broken | Functional |
| **Context** | None | Maintained |
| **Usability** | Completely broken | Fully functional |

---

#### **Lesson Learned**

This bug demonstrates:

1. **Off-by-one logic errors can be catastrophic** - A simple `<=` vs `>` destroyed the entire model

2. **Validation is critical** - Always test output after major operations

3. **Math matters** - The difference between 5% and 95% is the difference between working and broken

4. **Edge cases hide bugs** - The `if k <= 0` was meant for edge cases but was backwards

5. **Threshold logic is subtle** - "Keep top k" vs "prune bottom k" requires careful thinking

**Prevention:**
- Always validate with assertions: `assert 0.04 < actual_sparsity < 0.06`
- Test with small examples before full model
- Check intermediate values during development
- Use clear variable names: `num_to_keep` instead of `k`

---

**Solution:**
✅ **This bug has been fixed in the current code!**

The threshold logic now correctly keeps 95% and prunes 5% of parameters.

---

### Problem 2: Model Generates Gibberish

**Symptoms:**
```
Output: "asdkfj lkqwjelkj qwlekj qlwkej"
```

**Possible Causes:**
- Sparsity too high (removed too many parameters)
- Calibration data doesn't match your use case
- Model corrupted during save/load

**Solutions:**
1. **Reduce sparsity:** Try `SPARSITY = 0.03` (3%) instead of 0.05
2. **Use more calibration samples:** Increase `CALIBRATION_SIZE = 200`
3. **Check calibration data:** Ensure Java code samples are high quality
4. **Reload and re-prune:** Start fresh from the original model

---

### Problem 3: Out of Memory Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use smaller batch size during calibration:**
   ```python
   max_length=128  # Instead of 256
   ```

2. **Use CPU for pruning calculations:**
   ```python
   flat = importance.flatten().cpu()  # Move to CPU
   ```

3. **Process fewer samples:**
   ```python
   CALIBRATION_SIZE = 40  # Instead of 80
   ```

---

### Problem 4: Very Slow Pruning

**Symptoms:**
- Takes hours instead of minutes
- Progress bar stuck

**Solutions:**
1. **Ensure GPU is being used:**
   ```python
   print(model.device)  # Should show 'cuda:0' not 'cpu'
   ```

2. **Reduce calibration samples:**
   ```python
   CALIBRATION_SIZE = 50
   ```

3. **Use smaller max_length:**
   ```python
   truncation=True, max_length=128
   ```

---

### Problem 5: Sparsity Not Exactly 5%

**Symptoms:**
```
Pruned 4.87% instead of 5.00%
```

**Explanation:**
- This is normal and expected
- Pruning is applied per-layer independently
- Some layers might have slightly different sparsity
- Overall average should be close to target

**What to check:**
- If actual sparsity is 2-8%: ✅ Fine
- If actual sparsity is <1% or >10%: ❌ Something is wrong

---

## Summary

### What We Accomplished:

1. **Loaded** a 3-billion parameter language model
2. **Calibrated** it using 80 Java code samples
3. **Measured** which parameters are important for Java coding
4. **Pruned** the least important 5% of parameters (150 million)
5. **Saved** a more efficient version of the model
6. **Validated** that it still generates good code

### Key Takeaways:

- **Pruning makes models smaller and faster** without major quality loss
- **Wanda is smart** because it considers both weight size AND actual usage
- **Calibration is crucial** - it tells us which parameters matter for our specific task (Java coding)
- **The six layer types** (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj) contain most parameters
- **Proper threshold calculation** is critical - a bug here can destroy the model

### Next Steps:

1. **Test thoroughly:** Try the pruned model on various Java coding tasks
2. **Compare performance:** Benchmark speed and quality vs. original model
3. **Experiment with sparsity:** Try 3%, 7%, 10% to find the sweet spot
4. **Try different data:** Use Python, JavaScript, or other languages for calibration
5. **Measure improvements:** Track file size, inference speed, and memory usage

---

## Glossary

- **Activation:** How much a parameter "fires" or gets used when processing data
- **Calibration:** The process of measuring parameter importance using sample data
- **LLM:** Large Language Model - an AI system that understands and generates text
- **MLP:** Multi-Layer Perceptron - the "thinking" component that processes information
- **Parameter:** A single adjustable number in the model (like a knob)
- **Pruning:** Removing unimportant parameters to make the model more efficient
- **Sparsity:** The percentage of parameters that are zero (pruned)
- **Tensor:** A multi-dimensional array of numbers (like a spreadsheet with many sheets)
- **Tokenizer:** Converts text to numbers and vice versa
- **Wanda:** "Weights AND Activations" - a pruning technique that considers both factors
- **Weight:** The value of a parameter (the setting of the knob)

---

## References

- **Wanda Paper:** "A Simple and Effective Pruning Approach for Large Language Models"
- **Model:** Qwen2.5-Coder-3B-Instruct by Alibaba Cloud
- **Framework:** Hugging Face Transformers, PyTorch
- **Dataset:** CodeSearchNet (Java subset)

---

**Created:** December 2025
**Last Updated:** December 2025
**Purpose:** Educational guide for understanding LLM pruning
