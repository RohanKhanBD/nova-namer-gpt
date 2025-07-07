# BavCityGPT 🏔️

A lightweight character-level transformer model (6.35M parameters) for generating authentic Bavarian city names. This GPT implementation learns from a unique dataset combining real Bavarian settlements and natural landmarks to create novel, culturally authentic toponyms.

## 🎯 Key Features

- **Lightweight Architecture**: 6.35M parameter transformer optimized for character-level generation
- **Fast Training**: ~24 minutes on Apple Silicon (M1/M2)
- **Minimal Dependencies**: Only PyTorch and NumPy required
- **Complete Pipeline**: From data preparation to inference

## 📊 Model Performance

Best model achieved:
- **Training Loss**: 1.202 (NLL)
- **Validation Loss**: 1.441 (NLL)
- **Convergence**: 6000 iterations

## Sample examples
Selected novel place names inferenced from the best performing model:

- Untergammering
- Kammerlach
- Hinterroningen
- Schönkrippen
- Koppellerwiesmühle
- Eisenried
- Kirschensur
- Buchschönmühl
- Scharmannshausen
- Vogeltrück

## Model Architecture

BavCityGPT is build on state-of-the-art transformer tech with:
- **Character-level tokenization** for fine-grained linguistic patterns
- **8-layer transformer** with 8 attention heads and 256 embedding dimensions
- **6.35M parameters** optimized for Bavarian toponymy
- **64-token context window** enabling complex name pattern recognition
- **Few dependencies** Pytorch, Numpy

## Dataset: The Bavarian Blend

The training dataset represents a novel approach to place name generation, combining ~60,000 entries from multiple sources:

### Real Places (~45k entries)
- **Cities and villages** from official registries / APIs
- **Multiple administrative affiliations** creating natural repetitions that strengthen common patterns
- **Historical and modern settlements** across all Bavarian regions

### Natural Landmarks (~15k entries) 
- **Mountains, rivers, forests, and lakes** with authentic Bavarian nomenclature
- **Cleaned toponymic suffixes** (removed "-berg", "-see", "-wald") to prevent overfitting while preserving linguistic roots
- **Unique prefixes and stems** that capture Bavaria's diverse geographical heritage

This unprecedented blend enables the model to generate names that sound authentically Bavarian while creating entirely novel combinations

The raw dataset & metadata are in /data.


## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bavarian-city-gpt.git
cd bavarian-city-gpt
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Generate Names with Demo model

```bash
python sample.py
```
- This will generate 50 Bavarian city names using the pre-trained demo model.
- The samples will be printed and saved at .txt file into a samples folder in the model dir.
---

## Usage

### Basic Sampling

```bash
# Generate 50 names (default)
python sample.py

### Advanced Usage

```bash
# Use a specific model
python sample.py saved_models/your_model_directory

--- 

## Training Your Own Model

### 1. Prepare Data

Place your text data in `data/names.txt` (one name per line):

```
Burgsinn
Leitenthal
Maisenberg
Weidensorger
...
```

### 2. Process Data

```bash
python prepare_data.py
```

This creates binary files for efficient training.

### 3. Configure Model

Edit the configuration in `model.py`:

```python
@dataclass
class GPTconfig:
    context_len: int = 64  
    vocab_size: int = 61
    n_embd: int = 256
    n_head: int = 8
    # ... other settings
```

### 3. Configure Training

Edit the configuration in `config.py`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    train_iter: int = 10000        
    device: str = "mps"
    # ... other settings
```

### 4. Train Model

```bash
python train.py
```

The script uses settings from `TrainConfig` in `config.py`.

### 5. Sample from Your Model

```bash
python sample.py --out_dir saved_models/bavGPT_YYYYMMDD_HHMMSS
```

## 📁 Project Structure

```
bavcityGPT/
├── model.py          # Transformer architecture
├── train.py          # Training pipeline
├── sample.py         # Inference engine
├── prepare.py        # Data processing
├── config.py         # Configuration dataclasses
├── pyproject.toml    # Package configuration
├── data/
│   ├── names.txt     # Raw input data
│   └── *.bin         # Processed binary files
└── saved_models/
    └── demo/         # Pre-trained demo model
```

## Code Structure

BavCityGPT follows a modular design with clean separation of concerns:

#### Data Processing (`prepare.py`)
- **NameProcessor class** handles raw data ingestion and preprocessing
- **Character-level encoding/decoding** with vocabulary building
- **Train/dev/test splitting** with binary serialization for fast loading

#### Model Implementation (`model.py`)
- **GPT class** implements the core transformer architecture
- **Modular attention** with separate Head and MultiHeadAttention classes
- **Proper weight initialization** following GPT-2 standards

#### Training Pipeline (`train.py`)
- **NameGPTTrainer** orchestrates the complete training workflow
- **ModelManager** handles model persistence and metadata tracking
- **TrainingMetrics** provides loss evaluation and progress logging

#### Inference Engine (`sample.py`)
- **NameGPTSampler** supports both standalone and post-training generation
- **Temperature-controlled sampling** for creativity vs. coherence trade-offs
- **Automatic file management** with timestamped output

#### Configuration Management (`config.py`)
- **Dataclass-based configs** for training, sampling, and data processing
- **Environment-aware device selection** (MPS/CUDA/CPU)

### Workflow Integration
Raw Data → NameProcessor → Binary Files → NameGPTTrainer → Saved Model → NameGPTSampler → Generated Names

---

## Todos
- "inference-check" service which is called by sample.py to check if new generated name is part of dataset to avoid alreay existing names
- testcases
- Try different model / context_len approach with "one name within context padded to fixed len & special start and end chars"

## 🙏 Acknowledgments

- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Training data sourced from public Bavarian geographical databases
- Built with PyTorch


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## Contact
Feel free to reach out for collaboration or questions:

https://github.com/kenobijr

[mail](mailto:22.scree_rhino@icloud.com)
