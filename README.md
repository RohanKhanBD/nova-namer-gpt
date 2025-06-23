# BavCityGPT 🏔️

A state-of-the-art character-level 6.35M transformer model for generating authentic Bavarian City names. This GPT implementation creates novel names by learning from a unique blend of real Bavarian cities, villages, and natural landmarks. 

---

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

---

## Model Architecture

BavCityGPT is build on state-of-the-art transformer tech with:
- **Character-level tokenization** for fine-grained linguistic patterns
- **8-layer transformer** with 8 attention heads and 256 embedding dimensions
- **6.35M parameters** optimized for Bavarian toponymy
- **64-token context window** enabling complex name pattern recognition
- **Few dependencies** Pytorch, Numpy

---

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

This unprecedented blend enables the model to generate names that sound authentically Bavarian while creating entirely novel combinations—something no existing model has achieved.

The raw dataset & metadata are in /data.

---

## Performance

The best-performing model achieved NLLLoss:
- **Training Loss**: 1.202
- **Validation Loss**: 1.446
- **Training Time**: 23.86 minutes on Apple Silicon
- **Convergence**: Stable learning over 6,500 iterations

---

## Setup / Quick start
tbd

---

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

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact
Feel free to reach out for collaboration or questions:

https://github.com/kenobijr

[mail](mailto:22.scree_rhino@icloud.com)

