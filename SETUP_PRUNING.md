# Setup Guide for Running java_llm_prune.py

This guide will help you set up a new computer to run the LLM pruning script.

## Prerequisites

- A computer with a CUDA-compatible GPU (recommended for running the 3B model)
- At least 16GB RAM
- ~10GB free disk space for the model and dependencies
- Python 3.10 or later installed on your system

## Setup Options

Choose one of the following setup methods:
- **Option A: Using Python Virtual Environment (venv)** - Lightweight, built into Python
- **Option B: Using Conda** - More comprehensive environment management

---

## Option A: Setup with Python Virtual Environment (venv)

### Step 1: Create Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
# Navigate to project directory
cd /path/to/Pruning-LLM

# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
# For Linux/Mac:
source venv/bin/activate

# For Windows:
venv\Scripts\activate
```

### Step 2: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 3: Install PyTorch with CUDA Support

Install PyTorch with CUDA support (adjust CUDA version based on your system):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended, will be very slow)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Requirements

Install the required packages from requirements_for_prune.txt:

```bash
pip install -r requirements_for_prune.txt
```

This will install:
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- tqdm >= 4.65.0

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** To deactivate the virtual environment when you're done, simply run:
```bash
deactivate
```

---

## Option B: Setup with Conda

### Step 1: Install Conda

If you don't have Conda installed, download and install Miniconda or Anaconda:

**For Linux/Mac:**
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts and restart your terminal
```

**For Windows:**
Download from: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Create Conda Environment

Create a new conda environment with Python 3.10 or later:

```bash
conda create -n llm-prune python=3.10 -y
conda activate llm-prune
```

### Step 3: Install PyTorch with CUDA Support

Install PyTorch with CUDA support (adjust CUDA version based on your system):

```bash
# For CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU only (not recommended, will be very slow)
conda install pytorch cpuonly -c pytorch -y
```

### Step 4: Install Requirements

Install the required packages from requirements_for_prune.txt:

```bash
pip install -r requirements_for_prune.txt
```

This will install:
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- tqdm >= 4.65.0

---

## Common Steps (After Environment Setup)

### Step 1: Prepare Your Data

The script expects calibration data in JSONL format with a "code" field.

**Important:** You need to update the data paths in `java_llm_prune.py` at lines 22-24:

```python
dataset = load_dataset("json", data_files={
    "train": "path/to/your/train_small.jsonl",
    "validation": "path/to/your/train_small.jsonl",
    "test": "path/to/your/test_small.jsonl"
})
```

Your JSONL files should have entries like:
```json
{"code": "public class Example { ... }"}
{"code": "public void method() { ... }"}
```

### Step 2: Run the Pruning Script

Navigate to the notebooks directory and run:

```bash
python java_llm_prune.py
```

The script will:
1. Download the Qwen/Qwen2.5-Coder-3B-Instruct model (~6GB)
2. Load calibration data (80 samples by default)
3. Collect activation statistics
4. Apply Wanda pruning (5% sparsity by default)
5. Save the pruned model to `./QwenCoder3B_JavaPruned-5/`

---

## Configuration Options

You can modify these settings at the top of `java_llm_prune.py`:

- `MODEL_NAME`: The base model to prune (default: "Qwen/Qwen2.5-Coder-3B-Instruct")
- `SPARSITY`: Percentage of weights to prune (default: 0.05 = 5%)
- `CALIBRATION_SIZE`: Number of samples to use for calibration (default: 80)
- `SAVE_DIR`: Where to save the pruned model (default: "./QwenCoder3B_JavaPruned-5")

## Troubleshooting

### Out of Memory Error
- Reduce `CALIBRATION_SIZE` to a smaller number
- Use a machine with more GPU memory
- Modify the script to use CPU offloading (slower)

### Model Download Issues
- Ensure you have a stable internet connection
- The model download is ~6GB and may take time
- Check Hugging Face's status if downloads fail

### Import Errors
- Make sure you activated your environment:
  - For venv: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
  - For Conda: `conda activate llm-prune`
- Verify all packages installed correctly: `pip list`

## Expected Output

The script will display:
1. Model loading progress
2. Calibration data statistics
3. Number of targeted layers
4. Activation collection progress bar
5. Pruning statistics (number of parameters pruned)
6. Model save confirmation

## Next Steps

After pruning, you can:
- Test the pruned model's performance
- Convert to GGUF format for deployment
- Compare with the original model
- Experiment with different sparsity levels
