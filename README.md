# Trias: an encoder-decoder model for generating synthetic eukaryotic mRNA sequences

Trias is an encoder-decoder language model trained to reverse-translate protein sequences into codon sequences. It learns codon usage patterns from 10 million mRNA coding sequences across 640 vertebrate species, enabling context-aware sequence generation without requiring handcrafted rules.

<p align="center">
  <img src="overview.png" alt="Model Overview" width="700"/>
</p>


Trias is developed and tested with **Python 3.8.8**. 

## Dataset format

To train Trias, your dataset must include the following columns:
- `protein`: Amino acid sequence, must end with * (stop codon)
- `species_name`: Label identifying the species (e.g., "Homo sapiens")
- `mrna`: Full mRNA sequence
- `codon_start`: 0-based start index of coding region within mrna
- `codon_end`: 0-based end index (exclusive) of coding region within mrna

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
