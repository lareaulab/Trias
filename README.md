# Trias: an encoder-decoder model for generating synthetic eukaryotic mRNA sequences

Trias is an encoder-decoder language model trained to reverse-translate protein sequences into codon sequences. It learns codon usage patterns from 10 million mRNA coding sequences across 640 vertebrate species, enabling context-aware sequence generation without requiring handcrafted rules.

<p align="center">
  <img src="overview.png" alt="Model Overview" width="700"/>
</p>


## Setup and installation

Trias is developed and tested with **Python 3.8.8** and uses [Weights & Biases](https://docs.wandb.ai/quickstart/) for logging training progress.

We recommend using `conda`
```bash
conda create -n trias python=3.8.8
conda activate trias
```

Install dependencies
```bash
git clone https://github.com/lareaulab/Trias.git
cd Trias
pip install -e .
```
Or use `requirements.txt`
```bash
pip install -r requirements.txt
```


## Reverse Translation

Trias generates optimized codon sequences from protein input using a pretrained model. You can use the checkpoint hosted on Hugging Face (lareaulab/Trias) or a local model directory. It supports execution on both CPU and GPU (automatically detected). And we provide both greedy decoding and beam search for flexible output control.

Greedy decoding selects the most likely token at each step, it's faster and deterministic. Beam search explores multiple candidate paths and is better for longer or complex proteins, but is also slower.

Greedy search
```bash
python scripts/reverse_translation.py \
  --model_path lareaulab/Trias \
  --protein_sequence "MTEITAAMVKELRESTGAGMMDCKNALSETQ*" \
  --species "Homo sapiens" \
  --decoding greedy
```

Beam search
```bash
python scripts/reverse_translation.py \
  --model_path lareaulab/Trias \
  --protein_sequence "MTEITAAMVKELRESTGAGMMDCKNALSETQ*" \
  --species "Homo sapiens" \
  --decoding beam \
  --beam_width 5
```

## Dataset format

To train Trias, your dataset must include the following columns:
- `protein`: Amino acid sequence, must end with * (stop codon)
- `species_name`: Label identifying the species (e.g., "Homo sapiens")
- `mrna`: Full mRNA sequence
- `codon_start`: 0-based index of the first nucleotide of the coding region in the mrna
- `codon_end`: 0-based index of the last nucleotide of the stop codon

Supported file formats:
- `.parquet`, `.csv`, `.json` (auto-detected)


## Model training

Use the provided training script to launch a run
```bash
bash scripts/train_trias.sh
```
This launches a full training session using main.py. You can customize:

- Model architecture (hidden size, number of layers, attention heads, etc.)
- Training parameters (steps, batch size, learning rate, etc.)


## Citation

If you use Trias, please cite our work:

```bibtex
@article{faizi2025,
  title={A generative language model decodes contextual constraints on codon choice for mRNA design},
  author={Marjan Faizi and Helen Sakharova and Liana F. Lareau},
  journal={bioRxiv},
  year={2025},
  url={https://doi.org/10.1101/2025.05.13.653614}
}
```
