# LLM Pruning Procedure

## The Wanda Pruning Technique
### Step 1: Understanding Importance
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

### The Two Main Components will be pruned
#### 1. Attention Mechanism (The "Focus System")()
#### 2. MLP Layers (The "Thinking System")

### Visualization of One Layer

```
Input Text: "Write a Java function"
         ↓
┌──────────────────────────────────── ┐
│     ATTENTION MECHANISM             │
│  ┌────────────────────────────── ┐  │
│  │ Q_PROJ (What am I looking for?)│ │
│  │ K_PROJ (What is each word?)   │  │
│  │ V_PROJ (Word meanings)        │  │
│  │ O_PROJ (Combine results)      │  │
│  └────────────────────────────── ┘  │
└──────────────────────────────────── ┘
         ↓
┌────────────────────────────────────┐
│     MLP (THINKING)                 │
│  ┌──────────────────────────────┐  │
│  │ UP_PROJ (Expand for thinking) │ │
│  │ Processing...                 │ │
│  │ DOWN_PROJ (Compress results)  │ │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         ↓
Output: Generates next word
```

### Stage 1: Load Model & Data


#### Initialize model, sparsity, size of test data and the location
```
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SPARSITY = 0.05  # 5% pruning
CALIBRATION_SIZE = 80
SAVE_DIR = "./QwenCoder3B_JavaPruned-5"
```
- **Model:** Qwen2.5-Coder-3B-Instruct
- **Sparsity:** 5% (we'll remove 5% of parameters)
- **Calibration samples:** 80 Java code examples
- **Data source:** Small dataset of Java functions from real projects
#### Load model
```
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval() # switch to evaluation mode
```
#### Load datasets, take the train dataset as calibration data
```
texts = [ex["code"] for ex in dataset["train"].select(range(min(CALIBRATION_SIZE, len(dataset["train"]))))]
print(f"Using {len(texts)} calibration samples")
```

### Stage 2: Collect Activations (Calibration)
#### Process:
We run 80 Java code samples through the model and watch which parameters activate (get used).

#### The Process:

1. **Define target layers and find target layers**
```
target_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
    target_modules = {}
    param_to_module = {}
```
2.**Create 1 dict to store modules name and layer object, 1 dict for modules and the parameter**
```
for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(k in name for k in target_keywords):
            target_modules[name] = module
            param_to_module[f"{name}.weight"] = name
```

2. **Hook Installation(install the sensors on eachlayer):**
   - These sensors measure how much each parameter activates
   - They don't change anything, just observe
```
def make_hook(layer_name):
        def hook(module, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                act = inp[0].detach().reshape(-1, inp[0].size(-1)).abs().mean(dim=0).cpu()
                activations[layer_name].append(act)
        return hook

    hooks = [module.register_forward_hook(make_hook(name)) for name, module in target_modules.items()]
```

3. **Feed Data:**
   - Send all 80 Java code samples through the model
   - For each sample, the sensors record which parameters activate strongly
   ```
   print("Collecting activations...")
    for text in tqdm(texts, desc="Processing"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            model(**inputs)
    ```
4. **Remove Hook:**
    ```
    for h in hooks:
        h.remove()
    ```


5. **Calculate Average Activations:**
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

#### Why This Matters:
Without this step, we'd only know the **weight sizes** but not which parameters are **actually used** for Java code. This calibration ensures we keep Java-relevant parameters.

### Stage 3: Prune the Model (Wanda Algorithm)
This is the main part

#### For Each Layer:

#### 1. **Calculate importance**
```
For each parameter in the layer:
  Importance = |Weight| × Activation
```

**Example:**
- Parameter A: Weight = 2.5, Activation = 0.8 → Importance = 2.0
- Parameter B: Weight = 0.1, Activation = 0.3 → Importance = 0.03
- Parameter C: Weight = 5.0, Activation = 0.05 → Importance = 0.25

#### 2. **Find threshold**
1. Flatten all importance scores into one long list
2. Sort them from highest to lowest
3. Find the value at the 95th percentile (this is our threshold)

#### 3. **Create mask**
```
For each parameter:
  If importance >= threshold:
    mask = 1 (KEEP)
  Else:
    mask = 0 (REMOVE)
```
#### 4: Apply Mask

New Weight = Old Weight × Mask
```
mask = importance >= threshold
param.data *= mask
```

