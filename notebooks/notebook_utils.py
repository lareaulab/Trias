import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import random
from Bio.Seq import Seq
from RNA import fold
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import seaborn as sns
from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonPrediction import tokenize, validate_and_convert_organism


# Amino acid to codon dictionary
aa_to_codon = { 'A': ['GCA', 'GCC', 'GCG', 'GCT'],
                'R': ['AGA', 'AGG', 'CGA', 'CGC', 'CGG', 'CGT'],
                'N': ['AAC', 'AAT'],
                'D': ['GAC', 'GAT'],
                'C': ['TGC', 'TGT'],
                'Q': ['CAA', 'CAG'],
                'E': ['GAA', 'GAG'],
                'G': ['GGA', 'GGC', 'GGG', 'GGT'],
                'H': ['CAC', 'CAT'],
                'I': ['ATA', 'ATC', 'ATT'],
                'L': ['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG'],
                'K': ['AAA', 'AAG'],
                'M': ['ATG'],
                'F': ['TTC', 'TTT'],
                'P': ['CCA', 'CCC', 'CCG', 'CCT'],
                'S': ['AGC', 'AGT', 'TCA', 'TCC', 'TCG', 'TCT'],
                'T': ['ACA', 'ACC', 'ACG', 'ACT'],
                'W': ['TGG'],
                'Y': ['TAC', 'TAT'],
                'V': ['GTA', 'GTC', 'GTG', 'GTT'],
                '*': ['TAA', 'TAG', 'TGA']
                }
# Dictionary mapping codons and amino acids
codon_to_aa = {
    "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "TAT": "Tyr", "TAC": "Tyr", "TAA": "Stop", "TAG": "Stop",
    "TGT": "Cys", "TGC": "Cys", "TGA": "Stop", "TGG": "Trp",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "CAT": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "ATG": "Met",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "AAT": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "GAT": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

## Dictionary mapping amino acids to chemical properties
# source: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html
amino_acid_properties = {
    "F": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aromatic", "hydropathy": "Hydrophobic", "volume": "Very large"},
    "L": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Hydrophobic", "volume": "Large"},
    "S": {"polarity": "Polar", "charge": "Neutral", "chemical": "Hydroxyl", "hydropathy": "Neutral", "volume": "Very small"},
    "Y": {"polarity": "Polar", "charge": "Neutral", "chemical": "Aromatic", "hydropathy": "Neutral", "volume": "Very large"},
    "C": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Sulfur", "hydropathy": "Hydrophobic", "volume": "Small"},
    "W": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aromatic", "hydropathy": "Hydrophobic", "volume": "Very large"},
    "P": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Neutral", "volume": "Small"},
    "H": {"polarity": "Polar", "charge": "Positive", "chemical": "Basic", "hydropathy": "Neutral", "volume": "Medium"},
    "Q": {"polarity": "Polar", "charge": "Neutral", "chemical": "Amide", "hydropathy": "Hydrophilic", "volume": "Medium"},
    "R": {"polarity": "Polar", "charge": "Positive", "chemical": "Basic", "hydropathy": "Hydrophilic", "volume": "Large"},
    "I": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Hydrophobic", "volume": "Large"},
    "M": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Sulfur", "hydropathy": "Hydrophobic", "volume": "Large"},
    "T": {"polarity": "Polar", "charge": "Neutral", "chemical": "Hydroxyl", "hydropathy": "Neutral", "volume": "Small"},
    "N": {"polarity": "Polar", "charge": "Neutral", "chemical": "Amide", "hydropathy": "Hydrophilic", "volume": "Small"},
    "K": {"polarity": "Polar", "charge": "Positive", "chemical": "Basic", "hydropathy": "Hydrophilic", "volume": "Small"},
    "V": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Hydrophobic", "volume": "Medium"},
    "A": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Hydrophobic", "volume": "Very small"},
    "D": {"polarity": "Polar", "charge": "Negative", "chemical": "Acidic", "hydropathy": "Hydrophilic", "volume": "Small"},
    "E": {"polarity": "Polar", "charge": "Negative", "chemical": "Acidic", "hydropathy": "Hydrophilic", "volume": "Medium"},
    "G": {"polarity": "Nonpolar", "charge": "Neutral", "chemical": "Aliphatic", "hydropathy": "Neutral", "volume": "Very small"},
    "*": {"polarity": "Stop", "charge": "Stop", "chemical": "Stop", "hydropathy": "Stop", "volume": "Stop"},
}

amino_acid_properties_esm = {
    "F": "Aromatic",
    "L": "Hydrophobic",
    "S": "Polar",
    "Y": "Aromatic",
    "C": "Unique",
    "W": "Aromatic",
    "P": "Unique",
    "H": "Polar and \nPositively Charged",
    "Q": "Polar",
    "R": "Positively Charged",
    "I": "Hydrophobic",
    "M": "Hydrophobic",
    "T": "Polar",
    "N": "Polar",
    "K": "Positively Charged",
    "V": "Hydrophobic",
    "A": "Hydrophobic",
    "D": "Negatively Charged",
    "E": "Negatively Charged",
    "G": "Unique",
    "*": "Stop"
}

three_to_one_letter = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E',
    'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F',
    'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'Stop': '*'
}
one_to_three_letter = {v: k for k, v in three_to_one_letter.items()}


def generate_prompt_from_dataset(dataset):
    prompt = f">>{dataset['species_name']}<< {dataset['protein']}"
    return prompt

def seq_to_ids(seq, tokenizer, bos_eos=True):
    if bos_eos:
        codons = ["</s>"] + [seq[i:i+3] for i in range(0, len(seq), 3)] + ["</s>"] 
    else:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    target_codon_ids = torch.tensor([tokenizer._convert_token_to_id(c) for c in codons])
    return target_codon_ids

def calculate_gc_content(sequence):
    # Count the number of 'G' and 'C' nucleotides
    gc_count = sequence.count('G') + sequence.count('C')
    # Calculate GC content as a percentage
    gc_content = (gc_count / len(sequence)) * 100
    
    return gc_content

def calculate_gc3_content(sequence):
    sequence = sequence.upper()
    # Extract 3rd codon positions 
    third_positions = sequence[2::3]
    if len(third_positions) == 0:
        return 0

    gc3_content = calculate_gc_content(third_positions)

    return gc3_content

def calculate_u_content(sequence):
    sequence = sequence.replace('T', 'U')
    u_count = sequence.count('U')
    u_content = (u_count / len(sequence)) * 100

    return u_content

def calculate_average_srscu(sequence, srscu_dict):
    """Mean sRSCU across codons of a CDS.
    Args:        sequence (str): The CDS sequence.
                 srscu_dict (dict): Mapping from codon (e.g. "AAA") to sRSCU value.
    """
    seq = sequence.upper().replace("U", "T")
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    values = [srscu_dict.get(c) for c in codons]
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else float("nan")

def calculate_mfe(sequence):
    sequence = sequence.replace('T', 'U')
    _, mfe = fold(sequence)
    return mfe

def compute_normalized_dtw(true_min_max, predicted_min_max):
    """
    Compute normalized DTW distance between true and predicted %MinMax values.

    Args:
        true_min_max (list or np.array): %MinMax values from the wild-type sequence.
        predicted_min_max (list or np.array): %MinMax values from the generated sequence.

    Returns:
        float: Normalized DTW distance.
    """
    # Ensure both inputs are 1D NumPy arrays
    true_min_max = np.array(true_min_max).flatten()
    predicted_min_max = np.array(predicted_min_max).flatten()

    # Create (index, value) pairs to retain positional information
    x = np.array(list(enumerate(true_min_max)))  # [(index, value), ...]
    y = np.array(list(enumerate(predicted_min_max)))  # [(index, value), ...]

    # Compute DTW distance
    avg_length = (len(true_min_max) + len(predicted_min_max)) / 2
    distance, _ = fastdtw(x, y, dist=euclidean)

    # Normalize by the average sequence length
    normalized_distance = distance / avg_length
    return normalized_distance

def compute_dtw_matrix(synthetic_profiles, gfp_profiles):
    """Computes the DTW matrix comparing synthetic sequences to GFP variants."""
    dtw_results = {}
    
    for syn_name, syn_profile in synthetic_profiles.items():
        dtw_results[syn_name] = {
            gfp_name: compute_normalized_dtw(syn_profile, gfp_profile)
            for gfp_name, gfp_profile in gfp_profiles.items()
        }

    # Convert to DataFrame for better readability
    dtw_df = pd.DataFrame(dtw_results).T  # Transpose to align with synthetic sequences
    return dtw_df

def generate_labels(data, examples, srscu_df, max_length_data=None, level="token"):
    """
    Generate labels for sequences or tokens based on the provided dataset.

    Args:
        data (dict): Dataset containing "species_name", "protein", "mrna", "codon_start", and "codon_end".
        examples (int): Number of examples to process.
        max_length_data (int, optional): Maximum length of tokens per sequence (e.g., amino acids or codons).
        level (str): "token" for token-level labels, "sequence" for sequence-level labels.
    
    Returns:
        dict: A dictionary containing the generated labels.
    """
    if level not in ["token", "sequence"]:
        raise ValueError("Unsupported level. Choose 'token' or 'sequence'.")

    labels = {}

    if level == "token":
        # Token-level labels
        labels["Species"] = [gen for gen, seq in zip(data["species_name"][:examples], data["protein"][:examples]) for _ in seq[:max_length_data]]
        labels["Gene"] = [gen for gen, seq in zip(data["gene_name"][:examples], data["protein"][:examples]) for _ in seq[:max_length_data]]
        labels["Amino Acid"] = [aa for seq in data["protein"][:examples] for aa in seq[:max_length_data]]
        labels["Codon"] = [seq[i:i+3] for seq in data["wildtype_seq"][:examples] for i in range(0, len(seq), 3)]
        labels["Polarity"] = [amino_acid_properties.get(aa, {}).get("polarity", None) for aa in labels["Amino Acid"]]
        labels["Charge"] = [amino_acid_properties.get(aa, {}).get("charge", None) for aa in labels["Amino Acid"]]
        labels["Chemical"] = [amino_acid_properties.get(aa, {}).get("chemical", None) for aa in labels["Amino Acid"]]
        labels["Hydropathy"] = [amino_acid_properties.get(aa, {}).get("hydropathy", None) for aa in labels["Amino Acid"]]
        labels["Properties"] = [amino_acid_properties_esm.get(aa, None) for aa in labels["Amino Acid"]]
        labels["Volume"] = [amino_acid_properties.get(aa, {}).get("volume", None) for aa in labels["Amino Acid"]]
        labels["GC content"] = [calculate_gc_content(codon) if codon else None for codon in labels["Codon"]]
        labels["sRSCU"] = [srscu_df.get(codon, None) for codon in labels['Codon']]
        # replace '*' with 'Stop'
        labels["Amino Acid"] = ["Stop" if aa == "*" else aa for seq in data["protein"][:examples] for aa in seq[:max_length_data]]

    elif level == "sequence":
        # Sequence-level labels
        labels["Species"] = data["species_name"][:examples]
        labels["Gene"] = data["gene_name"][:examples]
        labels["GC content"] = [calculate_gc_content(seq) for seq in data["wildtype_seq"][:examples]]
        labels["sRSCU"] = [calculate_average_srscu(seq, srscu_df) for seq in data["wildtype_seq"][:examples]]

    return labels


def extract_embeddings(model, tokenizer, input_sequences, layer=-1, exclude_first_n_tokens=0, 
                       bart_state="encoder", level="sequence", decoder_input=None):
    """
    Extract embeddings from a model.

    Args:
        model: The HuggingFace model (e.g., BART).
        tokenizer: The tokenizer corresponding to the model.
        input_sequences: List of input sequences (e.g., protein sequences).
        layer: The layer to extract embeddings from (-1 = last layer).
        exclude_first_n_tokens: Number of tokens to exclude from the start (useful for CLS tokens).
        bart_state: "encoder" or "decoder" for BART models.
        decoder_input: Input for the decoder (needed for extracting decoder embeddings).
        level: "sequence" for sequence-level embeddings, "token" for token-level embeddings.

    Returns:
        embeddings: A NumPy array of embeddings.
    """
    all_embeddings = []

    # Tokenize sequences
    for seq in input_sequences:
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False, return_token_type_ids=False)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, decoder_input_ids=decoder_input)
            if bart_state == "encoder":
                hidden_states = outputs.encoder_hidden_states[layer]  # Encoder hidden states
            elif bart_state == "decoder":
                hidden_states = outputs.decoder_hidden_states[layer]  # Decoder hidden states

        # Remove batch dimension
        hidden_states = hidden_states.squeeze(0).cpu().numpy()  # [seq_len, hidden_size]

        # Exclude first n tokens if specified
        if exclude_first_n_tokens > 0:
            hidden_states = hidden_states[exclude_first_n_tokens:]

        if level == "sequence":
            # Average hidden states across tokens for sequence-level embeddings
            all_embeddings.append(hidden_states.mean(axis=0))  # [hidden_size]
        elif level == "token":
            # Append token-level embeddings
            all_embeddings.append(hidden_states)  # [seq_len, hidden_size]

    # Stack embeddings for sequence-level or flatten for token-level
    all_embeddings = np.vstack(all_embeddings) 

    return all_embeddings


def reduce_dimensionality(embeddings, method='umap', random_state=42, init='pca', **kwargs):
    """
    Reduce the dimensionality of embeddings using PCA, UMAP, or t-SNE.

    Args:
        embeddings (numpy.ndarray): Input embeddings to reduce. Shape: [num_samples, embedding_dim]
        method (str): Dimensionality reduction method. Supported: 'pca', 'umap', 'tsne'. Default: 'umap'.
        random_state (int): Random state for reproducibility. Default: 42.
        init (str): Initialization for t-SNE ('pca', 'random'). Default: 'pca'.
        **kwargs: Additional arguments to pass to the dimensionality reduction method.

    Returns:
        numpy.ndarray: Reduced embeddings. Shape: [num_samples, reduced_dim]
    """
    if method == 'pca':
        reducer = PCA(random_state=random_state, **kwargs)
    elif method == 'umap':
        reducer = UMAP(random_state=random_state, **kwargs)
    elif method == 'tsne':
        reducer = TSNE(init=init, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are 'pca', 'umap', and 'tsne'.")


    return reducer.fit_transform(embeddings)


def plot_embeddings(embeddings, labels, label_name, figure_name=None):
    """
    Plot embeddings in 2D space with categorical high-contrast coloring or numerical gradient.
    
    Args:
        embeddings (numpy.ndarray): 2D embeddings (e.g., from t-SNE or UMAP).
        labels (dict): Dictionary with label arrays.
        label_name (str): Key in `labels` to use for coloring.
    """
    if label_name not in labels:
        raise ValueError(f"Label '{label_name}' not found in labels.")

    # Create DataFrame
    df = pd.DataFrame(embeddings, columns=["Dim 1", "Dim 2"])
    df[label_name] = labels[label_name]

    # Check label type
    if np.issubdtype(df[label_name].dtype, np.number):
        # Numerical labels
        cmap = "coolwarm"
        legend = False
    else:
        # Categorical labels
        stop_color = "#A9A9A9"
        all_labels = sorted([lab for lab in df[label_name].unique() if lab != "Stop"]) + ["Stop"]
        num_labels = len(all_labels)

        if num_labels <= 8:
            # Use purple-themed palette
            dark_purples = ["indigo", "mediumpurple", "hotpink", "mediumvioletred"]
            soft_purples = ["mediumorchid", "plum", "lightpink"]
            color_assignment = []
            red_idx, blue_idx = 0, 0
            for i in range(num_labels - 1):  # exclude Stop
                if i % 2 == 0:
                    color_assignment.append(dark_purples[red_idx % len(dark_purples)])
                    red_idx += 1
                else:
                    color_assignment.append(soft_purples[blue_idx % len(soft_purples)])
                    blue_idx += 1
        else:
            # Use tab20 from matplotlib
            tab20_colors = [to_hex(c) for c in cm.get_cmap('tab20').colors]
            color_assignment = tab20_colors[:num_labels - 1]

        color_assignment.append(stop_color)
        cmap = dict(zip(all_labels, color_assignment))
        legend = True

    plt.figure(figsize=(5., 2.5), dpi=300)
    scatter = sns.scatterplot(
        data=df,
        x="Dim 1",
        y="Dim 2",
        hue=label_name,
        palette=cmap if legend else cmap,
        alpha=0.9,
        s=5,
        legend=legend,
    )
  
    if legend:
        handles = [plt.scatter([], [], c=color, label=label, s=25) for label, color in cmap.items()]
        plt.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.,
            ncol=2 if len(cmap) > 10 else 1,
            title=''
        )
    else:
        norm = plt.Normalize(df[label_name].min(), df[label_name].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(label_name, rotation=270, labelpad=15)

    sns.despine()
    plt.tight_layout()
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if figure_name: plt.savefig(figure_name, format='pdf')
    plt.show()

def plot_attention_weights(model, tokenizer, sequence, device, attention_type="all", layer=None, head=None, figure_name=None):
    inputs = tokenizer(sequence, return_token_type_ids=False, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    
    encoder_attentions = outputs.encoder_attentions  # Encoder self-attention
    decoder_attentions = outputs.decoder_attentions  # Decoder self-attention
    cross_attentions = outputs.cross_attentions      # Encoder-Decoder attention
    
    # Helper function to plot attention
    def plot_attention(attentions, layer=None, head=None):
        # Aggregate attention weights
        if layer is not None and head is not None:
            attention = attentions[layer][0][head].detach().cpu().numpy()
        elif layer is not None:
            attention = torch.mean(attentions[layer][0], dim=0).detach().cpu().numpy()
        elif head is not None:
            attention = torch.mean(torch.stack([attn[0][head] for attn in attentions]), dim=0).detach().cpu().numpy()
        else:
            attention = torch.mean(torch.stack([torch.mean(attn[0], dim=0) for attn in attentions]), dim=0).detach().cpu().numpy()
        
        plt.figure(figsize=(3, 2.5), dpi=300)
        ax = sns.heatmap(attention, cmap='Purples', cbar=False)
        plt.xlabel('Input protein sequence')
        plt.ylabel('Output codon sequence')
        tick_positions = list(range(0, attention.shape[1], 10))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions, rotation=90)
        ax.set_yticklabels(tick_positions)
        plt.xticks(rotation=0)
        plt.tight_layout()
        if figure_name: plt.savefig(figure_name, format='pdf')
        plt.show()

    # Plot encoder self-attention
    if encoder_attentions and (attention_type=="encoder" or attention_type=="all"):
        plot_attention(encoder_attentions, layer, head)
    # Plot decoder self-attention
    if decoder_attentions and (attention_type=="decoder" or attention_type=="all"):
        plot_attention(decoder_attentions, layer, head)
    # Plot encoder-decoder attention
    if cross_attentions and (attention_type=="cross" or attention_type=="all"):
        plot_attention(cross_attentions, layer, head)


def calculate_mutational_distance(codon1, codon2):
    """Calculate the mutational distance between two codons."""
    if len(codon1) != len(codon2):
        return None
    
    distance = sum(1 for a, b in zip(codon1, codon2) if a != b)
    mismatch_positions = [i for i, (a, b) in enumerate(zip(codon1, codon2), start=1) if a != b]
    return distance, mismatch_positions

purines = ['A', 'G']
pyrimidines = ['C', 'T']


def classify_mutation(codon1, codon2):
    """Classify the mutation as a transition or transversion."""
    mutation_type = []
    for i in range(len(codon1)):
        if codon1[i] != codon2[i] :
            if codon1[i] in purines and codon2[i] in purines:
                mutation_type.append("Transition")
            elif codon1[i] in pyrimidines and codon2[i] in pyrimidines:
                mutation_type.append("Transition")
            else:
                mutation_type.append("Transversion")
    return mutation_type


def scatter_with_marginals(df, x, y, title, xlabel, ylabel, figure_dir='',
                           highlighted_genes=None, annotation_dict=None,
                           xlim=None, ylim=None, color='mediumpurple'):
    # Create a joint plot with scatter kind and custom settings
    g = sns.jointplot(x=x, y=y, data=df,
                      height=2.8, ratio=4, space=0.1,
                      color=color, alpha=0.7,
                      joint_kws={"s": 13, "edgecolor": "white", "linewidth": 0.2, "zorder": 3},
                      marginal_kws=dict(bins=30, fill=True)
                      )
    # Check if highlighting is needed
    if highlighted_genes is not None:
        # Create a column to highlight genes
        df['highlight'] = df['record_id'].isin(highlighted_genes)
        colors = df['highlight'].map({True: 'mediumorchid', False: 'rebeccapurple'})

        for coll in g.ax_joint.collections:
            coll.remove()
        g.ax_joint.scatter(df[x], df[y], s=13, alpha=0.7, c=colors, edgecolor='white', linewidth=0.2, zorder=3)

    # Optionally set axis limits
    if xlim: g.ax_joint.set_xlim(xlim)
    if ylim: g.ax_joint.set_ylim(ylim)

    # Draw a line of equality
    lims = [max(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
            min(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
    g.ax_joint.plot(lims, lims, ls="--", c=".3")

    # Add label annotations (e.g., *, †, ‡, ▲, ◆, ■)
    if annotation_dict:
        for record_id, symbol in annotation_dict.items():
            row = df[df['record_id'] == record_id]
            if not row.empty:
                x_val = row[x].values[0]
                y_val = row[y].values[0]
                g.ax_joint.text(x_val, y_val, symbol,
                                fontsize=5,
                                ha='center', va='center',
                                color='black')

    # Adjust the top space to accommodate the title
    g.set_axis_labels(xlabel, ylabel)
    plt.subplots_adjust(top=0.9)
    plt.savefig(figure_dir+title+'.pdf', format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def greedy_search(model, tokenizer, input, max_length=2046, device=None):
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
        
    model.to(device)
    input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
    greedy_output = model.generate(input_ids,
                                    max_length=max_length, 
                                    output_scores=True,
                                    return_dict_in_generate=True)
    generated_sequence = greedy_output.sequences[0]
    sequence_scores = greedy_output.scores
    return generated_sequence, sequence_scores


def calculate_min_max(sequence, codon_frequencies, codon_to_aa, window_size):
    """
    Calculate the %MinMax values for a codon sequence using the corrected algorithm and codon_to_aa mapping.
    
    Parameters:
    - sequence (list of str): Codon sequence (list of codons).
    - codon_frequencies (dict): Dictionary containing codon frequencies.
    - codon_to_aa (dict): Dictionary mapping codons to their corresponding amino acids.
    - window_size (int): Size of the sliding window.
    
    Returns:
    - tuple: (%MinMax, array of %Min values, array of %Max values)
    """
    if len(sequence) < window_size:
        raise ValueError("Sequence length must be greater than or equal to the window size.")
    
    # Group codons by amino acids
    aa_to_codons = {}
    for codon, aa in codon_to_aa.items():
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        aa_to_codons[aa].append(codon)
    
    # Calculate X_avg,i, X_max,i, and X_min,i for each codon
    codon_to_stats = {}
    for aa, codons in aa_to_codons.items():
        codon_freqs = [codon_frequencies.get(codon, 0) for codon in codons]
        avg_usage = np.mean(codon_freqs)
        max_usage = np.max(codon_freqs)
        min_usage = np.min(codon_freqs)
        for codon in codons:
            codon_to_stats[codon] = {"avg": avg_usage, "max": max_usage, "min": min_usage}

    # Convert codon sequence to their frequencies and statistics
    freq_values = np.array([codon_frequencies.get(codon, 0) for codon in sequence])
    avg_values = np.array([codon_to_stats[codon]["avg"] for codon in sequence])
    max_values = np.array([codon_to_stats[codon]["max"] for codon in sequence])
    min_values = np.array([codon_to_stats[codon]["min"] for codon in sequence])

    # Calculate %MinMax using a sliding window
    min_max_values = []
    # Number of windows to calculate
    num_windows = len(sequence) - window_size + 1

    for i in range(num_windows):
        # Extract the current window
        window_freq = freq_values[i:i + window_size]
        window_avg = avg_values[i:i + window_size]
        window_max = max_values[i:i + window_size]
        window_min = min_values[i:i + window_size]
        
        if window_freq.sum() > window_avg.sum():
            value = 100 * (np.sum(window_freq - window_avg) / np.sum(window_max - window_avg))
        else:
            value = - 100 * (np.sum(window_avg - window_freq) / np.sum(window_avg - window_min)) 

        min_max_values.append(value)
    
    return min_max_values


def seq_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3)]


def categorize_codon(wt_srscu, pred_srscu, srscu_threshold):
    if wt_srscu < srscu_threshold and pred_srscu < srscu_threshold:
        return "Correctly predicted \nrare codon"
    elif wt_srscu < srscu_threshold and pred_srscu >= srscu_threshold:
        return "Misclassified as \ncommon codon"
    elif wt_srscu >= srscu_threshold and pred_srscu < srscu_threshold:
        return "Misclassified as \nrare codon"
    else:
        return "Correctly predicted \ncommon codon"


def extract_codon_pairs(input_df):
    """
    Efficiently extracts codon pairs using vectorized operations.
    
    Returns:
        DataFrame with codon pairs, their classification, and count.
    """
    output_df = pd.DataFrame()
    output_df["Predicted Codon"] = input_df["Predicted Codon"] # True Codon
    output_df["Codon Category"] = input_df["Codon Category"]
    # Identify sequence start by detecting 'Codon Index == 0'
    output_df["New Sequence"] = input_df["Codon Index"] == 0
    # Create a column that shifts codons and categories by 1 (previous row)
    output_df["Prev Codon"] = input_df["Predicted Codon"].shift(1)
    # Remove invalid pairs at sequence boundaries
    output_df = output_df[~output_df["New Sequence"]]
    # Create codon pairs
    output_df["Codon Pair"] = output_df["Prev Codon"] + "-" + output_df["Predicted Codon"]
    # Count occurrences of each pair-category combination
    codon_pair_df = output_df.groupby(["Codon Pair", "Codon Category"]).size().reset_index(name="Count")

    return codon_pair_df


def get_codon_sequences(protein_sequence, rscu_table, codon_dist=None, strategy="highest"):
    """
    Generates a codon sequence for a given protein sequence based on a selection strategy.
    
    Args:
    - protein_sequence (str): Amino acid sequence.
    - rscu_table (dict): Dictionary mapping amino acids to their respective codon RSCU values.
                         Example: {"A": {"GCT": 0.8, "GCC": 1.2, "GCA": 0.9, "GCG": 1.1}, ...}
    - codon_dist (dict, optional): Dictionary mapping amino acids to codon usage frequencies.
                                   Example: {"A": {"GCT": 0.3, "GCC": 0.4, "GCA": 0.2, "GCG": 0.1}, ...}
    - strategy (str): Strategy for codon selection.
                      Options: "highest" (default), "lowest", "distribution"
    
    Returns:
    - codon_sequence (str): Codon sequence generated based on the chosen strategy.
    """
    codon_sequence = []
    
    for aa in protein_sequence:
        if aa in rscu_table:
            codons = rscu_table[aa]
            
            if strategy == "highest":
                selected_codon = max(codons, key=codons.get)  # Codon with max RSCU
            elif strategy == "lowest":
                selected_codon = min(codons, key=codons.get)  # Codon with min RSCU
            elif strategy == "distribution" and codon_dist:
                # Select codon based on frequency distribution
                codon_choices = list(codon_dist[aa].keys())
                probabilities = np.array(list(codon_dist[aa].values()))
                selected_codon = np.random.choice(codon_choices, p=probabilities)
            else:
                raise ValueError("Invalid strategy. Choose 'highest', 'lowest', or 'distribution'.")
            
            codon_sequence.append(selected_codon)
    
    return "".join(codon_sequence)


def constrained_decoding(model, tokenizer, prompt, codon_seq, device=None):
    """Forced-decoding pass: feed `codon_seq` as decoder_input_ids and return logits.

    Args:
        model:     Trained ``BartForConditionalGeneration``.
        tokenizer: Matching ``TriasTokenizer``.
        prompt:    Encoder input, e.g. ``">>Homo sapiens<< MAKT*"``.
        codon_seq: In-frame CDS to evaluate (T, not U).
        device:    Optional device override; defaults to the model's device.

    Returns:
        ``(generated_sequence, sequence_scores)`` — 1-D token-ID tensor (length T)
        and ``(T-1, vocab_size)`` logit tensor.
    """
    dev = device if device is not None else next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)
    codons = ["</s>"] + [codon_seq[i:i+3] for i in range(0, len(codon_seq), 3)] + ["</s>"]
    target_ids = torch.tensor(
        [[tokenizer._convert_token_to_id(c) for c in codons]], device=dev,
    )
    with torch.no_grad():
        out = model(input_ids=input_ids, decoder_input_ids=target_ids)
    return target_ids[0], out.logits[0, :-1, :]


def extract_probs_from_logits(logits, token_ids, device="cpu"):
    """ Extracts probabilities for the actual generated tokens from logits """
    # Ensure both tensors are on the same device
    logits = logits.to(device)
    token_ids = token_ids.to(device)
    
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    # Align token_ids with logits (truncate the last token ID)
    token_ids = token_ids[1:]  # Modified to handle batch dimension
    # Extract probabilities for the actual generated tokens
    token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_probs.cpu().numpy()


def neg_log_likelihood(sequence, scores):
    log_likelihood = 0.0
    for i, logits in enumerate(scores):
        probs = F.log_softmax(logits, dim=-1)
        token_id = sequence[i+1]  # +1 because the first token is the <s> token
        log_likelihood += probs[token_id].item()
    return abs(log_likelihood) / (len(sequence)-1)


def sequence_similarity(predicted_sequence, true_sequence, sequence_type):
    if sequence_type == "codon":
        predicted_sequence = [predicted_sequence[i:i+3] for i in range(0, len(predicted_sequence), 3)]
        true_sequence = [true_sequence[i:i+3] for i in range(0, len(true_sequence), 3)]
    elif sequence_type == "protein":
        predicted_sequence = str(Seq(predicted_sequence).translate())
        true_sequence = str(Seq(true_sequence).translate())

    # Truncate both sequences to the minimum length
    max_length = min(len(predicted_sequence), len(true_sequence))
    predicted_sequence = predicted_sequence[:max_length]
    true_sequence = true_sequence[:max_length]

    common_tokens = sum(1 for pt, tt in zip(predicted_sequence, true_sequence) if pt == tt)

    similarity = common_tokens / max_length if max_length > 0 else 0
    return similarity


def trias_score(cds_seqs, species, model, tokenizer, device=None):
    """Score CDS sequences with Trias via forced-decoding.

    For each CDS, runs the model with ``decoder_input_ids`` clamped to the
    sequence's codons and returns the geometric-mean per-token probability
    (a value in (0, 1] — higher = more natural under the model).

    Args:
        cds_seqs:  Iterable of coding DNA sequences (T, not U).
        species:   Species tag for the encoder prompt, e.g. "Homo sapiens".
        model:     Trained ``BartForConditionalGeneration``.
        tokenizer: Matching ``TriasTokenizer``.
        device:    Optional device override; defaults to the model's device.

    Returns:
        1-D ``np.ndarray`` of float scores, one per input sequence.
    """
    dev = device if device is not None else next(model.parameters()).device
    protein = str(Seq(cds_seqs[0]).translate())
    prompt = f">>{species}<< {protein}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev)

    scores = []
    for seq in cds_seqs:
        dna = seq.replace("U", "T")
        codons = ["</s>"] + [dna[i:i+3] for i in range(0, len(dna), 3)] + ["</s>"]
        target_ids = torch.tensor(
            [[tokenizer._convert_token_to_id(c) for c in codons]], device=dev,
        )
        with torch.no_grad():
            logits = model(input_ids=input_ids, decoder_input_ids=target_ids).logits[0, :-1, :]

        gen_seq = target_ids[0]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ids = gen_seq[1 : len(logits) + 1]
        token_logp = log_probs[torch.arange(len(logits), device=dev), token_ids]
        scores.append(float(np.exp(token_logp.sum().item() / (len(gen_seq) - 1))))

    return np.array(scores, dtype=float)

def gmm_cutoff(x, resolution=100):

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(x)

    xs = np.linspace(x.min(), x.max(), resolution).reshape(-1, 1)
    resp = gmm.predict_proba(xs)

    diff = np.abs(resp[:,0] - resp[:,1])
    thr = xs[np.argmin(diff)][0]

    return thr

@torch.no_grad()
def rare_vs_common_codon_probs(
    protein_seq, codons_highest, codons_lowest,
    model, tokenizer, species="Homo sapiens", device=None,
):
    """Per-position probability of rare vs common codon under Trias.

    Forced-decodes against `codons_lowest` and reads off the probability
    Trias assigns to the matched-rare codon vs the matched-common codon
    at each position. Returns two 1-D arrays (p_rare, p_common).
    """
    dev = device if device is not None else next(model.parameters()).device
    prompt = f">>{species}<< {protein_seq}"

    low_ids, seq_score = constrained_decoding(model, tokenizer, prompt, codons_lowest, device=dev)

    high_codons = ["</s>"] + [codons_highest[j:j+3] for j in range(0, len(codons_highest), 3)] + ["</s>"]
    high_ids = torch.tensor(
        [[tokenizer._convert_token_to_id(c) for c in high_codons]], device=dev,
    )

    p_rare   = extract_probs_from_logits(seq_score, low_ids,     device=dev)
    p_common = extract_probs_from_logits(seq_score, high_ids[0], device=dev)
    return p_rare, p_common


def codontx_score(cds_seqs, organism, model, tokenizer, device="cpu"):
    """Score CDS sequences with CodonTransformer.

    Returns the geometric-mean per-codon probability for each CDS — NaN for
    sequences whose length is not a multiple of 3 or that contain internal stops.

    Requires the ``CodonTransformer`` package.
    """

    @torch.no_grad()
    def _probs(protein, organism_id, dev):
        merged = get_merged_seq(protein=protein.lower(), dna="")
        enc = tokenize(
            [{"idx": 0, "codons": merged, "organism": organism_id}],
            tokenizer=tokenizer,
        ).to(dev)
        logits = model(**enc, return_dict=True).logits[:, 1:-1, :][:, :len(protein), :]
        return F.softmax(logits, dim=-1)

    organism_id, _ = validate_and_convert_organism(organism)
    scores = []
    for cds in cds_seqs:
        cds_clean = cds.upper().replace("U", "T")
        if len(cds_clean) % 3 != 0:
            scores.append(np.nan); continue
        protein = str(Seq(cds_clean).translate())
        if "*" in protein[:-1]:
            scores.append(np.nan); continue
        if protein.endswith("*"):
            protein = protein[:-1]

        codons = [cds_clean[i:i+3].lower() for i in range(0, 3 * len(protein), 3)]
        probs = _probs(protein, organism_id, device)

        bad = False
        logps = []
        for i, (aa, codon) in enumerate(zip(protein.lower(), codons)):
            tid = tokenizer.convert_tokens_to_ids(f"{aa}_{codon}")
            if tid is None or tid < 0:
                bad = True; break
            logps.append(np.log(probs[0, i, tid].item()))
        scores.append(float(np.exp(np.mean(logps))) if not bad else np.nan)
    return np.asarray(scores, dtype=float)


@torch.no_grad()
def decodon_score(cds_seqs, model_decodon, taxid=9606, device="cpu"):
    """Score CDS sequences with cdsFM DeCodon.

    Returns geometric-mean next-token probability for each CDS, conditioned on
    *taxid* (e.g. 9606 = Homo sapiens, 10090 = Mus musculus).
    """
    tokenizer = model_decodon.tokenizer
    model = model_decodon.model.to(device).eval()

    tax_token = f"<{int(taxid)}>"
    if hasattr(tokenizer, "encoder") and tax_token in tokenizer.encoder:
        tax_id = tokenizer.encoder[tax_token]
    else:
        tax_id = tokenizer.convert_tokens_to_ids(tax_token)
    if tax_id is None or tax_id < 0:
        raise ValueError(f"Taxid token {tax_token} not found in tokenizer.")

    scores = []
    for seq in cds_seqs:
        clean = seq.upper().replace("U", "T")
        clean = "".join(c for c in clean if c in "ACGT")
        if len(clean) < 3:
            scores.append(np.nan); continue

        auto_seq = f"{tokenizer.cls_token}{clean}{tokenizer.sep_token}"
        enc = tokenizer(
            auto_seq, return_tensors="pt", truncation=True,
            padding=False, add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        input_ids[:, 0] = tax_id

        logits = model(input_ids=input_ids, return_dict=True).logits
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        scores.append(torch.exp(token_logp.mean()).item())
    return np.asarray(scores, dtype=float)


# General function to generate sequences with different decoding strategies
def generate_sequence(
    model,
    tokenizer,
    input,
    max_length=2048,
    device="cpu",
    **gen_kwargs,
):
    model.to(device)
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        output_scores=True,
        return_dict_in_generate=True,
        **gen_kwargs,
    )
    generated_ids = output.sequences[0]
    scores = output.scores

    return generated_ids, scores



def baseline_predictor(tokenizer, protein_seq, species_name, codon_dist, mode="sample"):
    """
    Generate a codon sequence from a protein sequence using different strategies.
    
    mode options:
        - "sample": sample from background codon distribution (probabilistic)
        - "max": choose the codon with highest probability
        - "uniform": choose codon uniformly at random
    """
    generated_ids = []

    for aa in protein_seq:
        codons = aa_to_codon[aa]
        probs = np.array([codon_dist[c] for c in codons])

        probs = probs / probs.sum()  # normalize just in case

        if mode == "sample":
            chosen_codon = np.random.choice(codons, p=probs)
        elif mode == "max":
            chosen_codon = codons[np.argmax(probs)]
        elif mode == "uniform":
            chosen_codon = np.random.choice(codons)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from 'sample', 'max', 'uniform'.")

        generated_ids.append(tokenizer.get_vocab()[chosen_codon])

    codon_seq = tokenizer.convert_ids_to_tokens(generated_ids)
    return tokenizer.convert_tokens_to_string(codon_seq)