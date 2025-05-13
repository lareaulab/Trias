# Trias: an encoder-decoder model for generating synthetic eukaryotic mRNA sequences

Trias is an encoder-decoder language model trained to reverse-translate protein sequences into codon sequences. It learns codon usage patterns from 10 million mRNA coding sequences across 640 vertebrate species, enabling context-aware sequence generation without requiring handcrafted rules.

<p align="center">
  <img src="overview.png" alt="Model Overview" width="700"/>
</p>


Trias is developed and tested with **Python 3.8.8**. 

## Reverse Translation

Trias can generate optimized codon sequences from protein input using a pretrained model. You can use any checkpoint hosted on Hugging Face (e.g., lareaulab/Trias) or a local model directory. It supports execution on both CPU and GPU (automatically detected). And we provide both greedy decoding and beam search for flexible output control.

CPU example:
```bash
python scripts/reverse_translation.py \
  --model_path ../src/trias \
  --protein_sequence "MTEITAAMVKELRESTGAGMMDCKNALSETQ*" \
  --decoding greedy
```

GPU example (with beam search)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reverse_translation.py \
  --model_path ../src/trias \
  --protein_sequence "MTEITAAMVKELRESTGAGMMDCKNALSETQ*" \
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


## Running Trias
...


## Citation

If you use Trias, please cite our work:

```bibtex
@article{faizi2025,
  title={A generative language model decodes contextual constraints on codon choice for mRNA design},
  author={Marjan Faizi and Helen Sakharova and Liana F. Lareau},
  journal={bioRxiv},
  year={2025},
  url={https://doi.org/xxxx/xxxx}
}
```
