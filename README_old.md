# 🧠 ChunkedDecomp: Efficient KV Cache Compression for Transformers

**ChunkedDecomp** compresses the Key-Value (KV) cache in transformer models using a novel chunked matrix decomposition (A/B matrices) to reduce memory usage during inference.

---

## 🚀 What It Does

- 📦 Compresses KV cache using low-rank chunked SVD
- 💾 Reduces memory usage significantly
- 🔁 Reconstructs only needed chunks during inference
- 📉 Benchmarks compression vs quality tradeoffs
- 🧪 Evaluates performance (perplexity, BLEU, etc.)
- 💻 Works on local machines or HPC clusters (SLURM-ready)

---

## 📁 Folder Structure

chunked-decomp/
├── configs/ # YAML configs for compression + models
├── data/ # Place for dummy input or dataset files
├── notebooks/ # Jupyter notebooks for interactive analysis
├── results/ # Outputs from all scripts
├── scripts/ # CLI + SLURM scripts
│ └── cluster_scripts/ # SLURM job + environment setup scripts
├── src/ # Core code
│ ├── models/ # ChunkedDecomp logic, KV cache handler
│ ├── utils/ # SVD, memory, dataset helpers
│ └── evaluation/ # Evaluation + memory tracking
├── tests/ # Unit tests
├── requirements.txt # Pip-based install
├── environment.yml # Conda install (see below ✅)
├── setup.py # pip install -e . support
├── .gitignore
└── README.md

yaml
Copy
Edit

---

## ⚙️ Installation

### ✅ Option 1: Conda (Recommended)
```bash
git clone https://github.com/yourusername/chunked-decomp.git
cd chunked-decomp
conda env create -f environment.yml
conda activate chunked-decomp
<details> <summary><strong>📄 environment.yml (included)</strong></summary>
yaml
Copy
Edit
name: chunked-decomp
channels:
  - defaults
  - pytorch
  - conda-forge

dependencies:
  - python=3.9
  - numpy>=1.24.0
  - scipy>=1.10.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - tqdm>=4.65.0
  - pytorch>=2.0.0
  - torchvision
  - torchaudio
  - pip
  - pip:
      - transformers>=4.30.0
      - datasets>=2.10.0
      - accelerate>=0.20.0
      - wandb>=0.15.0
</details>
✅ Option 2: pip + venv
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
pip install -e .
✅ Quick Test
bash
Copy
Edit
python scripts/run_compression.py ^
  --model_name distilgpt2 ^
  --compression_ratio 0.5 ^
  --num_samples 100 ^
  --test_mode ^
  --output_dir results/test/
You should see:

Progress logs in terminal

Files created in results/test/

🔨 Full Usage
🔹 Run Compression
bash
Copy
Edit
python scripts/run_compression.py ^
  --model_name gpt2 ^
  --compression_ratio 0.5 ^
  --chunk_size 64 ^
  --output_dir results/compression/
🔹 Evaluate Model Quality
bash
Copy
Edit
python scripts/evaluate_model.py ^
  --model_name gpt2 ^
  --compression_ratio 0.5 ^
  --metrics perplexity bleu ^
  --dataset_name wikitext ^
  --output_dir results/evaluation/
🔹 Benchmark Memory
bash
Copy
Edit
python scripts/benchmark_memory.py ^
  --model_name gpt2 ^
  --sequence_lengths 256 512 1024 ^
  --batch_sizes 1 4 ^
  --compression_ratios 0.3 0.5 0.7 ^
  --output_dir results/memory/
🧪 Run Unit Tests
bash
Copy
Edit
python -m pytest tests/ -v
📊 Visualize with Jupyter
bash
Copy
Edit
jupyter notebook
# Open notebooks/exploration.ipynb
Includes:

Compression stats

Memory vs accuracy plots

Config tweaks and debugging

🧾 Dummy Input
Create data/sample_input.txt:

pgsql
Copy
Edit
The quick brown fox jumps over the lazy dog.
Hello! This is a test input.
In any script:

python
Copy
Edit
with open("data/sample_input.txt") as f:
    lines = f.readlines()
💻 HPC Usage (SLURM)
1. SSH to your cluster
bash
Copy
Edit
ssh username@cluster.edu
2. Clone and Set Up
bash
Copy
Edit
git clone https://github.com/yourusername/chunked-decomp.git
cd chunked-decomp
chmod +x scripts/cluster_scripts/setup_environment.sh
./scripts/cluster_scripts/setup_environment.sh
3. Submit a SLURM Job
bash
Copy
Edit
sbatch scripts/cluster_scripts/submit_job.sh
4. Monitor Job
bash
Copy
Edit
squeue -u yourusername
🛠️ Custom Configs
Use configs/compression_configs.yaml:

yaml
Copy
Edit
balanced:
  compression_ratio: 0.5
  chunk_size: 64
  adaptive_rank: true
  min_rank_ratio: 0.15
In code:

python
Copy
Edit
import yaml
from src import ChunkedDecomp

with open("configs/compression_configs.yaml") as f:
    cfg = yaml.safe_load(f)

compressor = ChunkedDecomp(**cfg["balanced"])
🧠 Troubleshooting
Problem	Fix
CUDA out of memory	Reduce batch size or sequence length
Low quality	Use compression_ratio=0.7+ or increase min_rank_ratio
Slow reconstruction	Increase chunk_size, enable use_fast_svd
Import errors	Run pip install -e . in root folder

🥇 Credits
Built by [Your Name]
If this helped your research, please consider citing the repo.

🛣️ Roadmap
 Add quantization support

 LoRA compatibility

 Online adaptation for streaming models

🎉 Summary
✅ Compress large transformer KV caches
✅ Run and benchmark locally or on HPC
✅ Evaluate memory savings and accuracy
✅ All with clear configs, logs, and outputs

Happy compressing! 🚀

yaml
Copy
Edit

---

Let me know if you want this auto-populated with your actual GitHub username, HPC path, or model preferences.