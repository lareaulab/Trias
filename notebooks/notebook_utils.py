

def generate_prompt_from_dataset(dataset):
    prompt = f">>{dataset['species_name']}<< {dataset['protein']}"
    return prompt


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

    human_srscu_df = srscu_df[srscu_df.index == 'Homo sapiens']

    labels = {}

    if level == "token":
        # Token-level labels
        labels["Species"] = [gen for gen, seq in zip(data["species_name"][:examples], data["protein"][:examples]) for _ in seq[:max_length_data]]
        labels["Gene"] = [gen for gen, seq in zip(data["gene_name"][:examples], data["protein"][:examples]) for _ in seq[:max_length_data]]
        labels["Amino Acid"] = [aa for seq in data["protein"][:examples] for aa in seq[:max_length_data]]
        labels["Codon"] = [seq[i:i+3]
                           for seq, start, end in zip(data["mrna"][:examples], data["codon_start"][:examples], data["codon_end"][:examples])
                           for i in range(start, min(end, start + 3 * max_length_data), 3)]
        labels["Polarity"] = [amino_acid_properties.get(aa, {}).get("polarity", None) for aa in labels["Amino Acid"]]
        labels["Charge"] = [amino_acid_properties.get(aa, {}).get("charge", None) for aa in labels["Amino Acid"]]
        labels["Chemical"] = [amino_acid_properties.get(aa, {}).get("chemical", None) for aa in labels["Amino Acid"]]
        labels["Hydropathy"] = [amino_acid_properties.get(aa, {}).get("hydropathy", None) for aa in labels["Amino Acid"]]
        labels["Properties"] = [amino_acid_properties_esm.get(aa, None) for aa in labels["Amino Acid"]]
        labels["Volume"] = [amino_acid_properties.get(aa, {}).get("volume", None) for aa in labels["Amino Acid"]]
        labels["GC content"] = [calculate_gc_content(codon) if codon else None for codon in labels["Codon"]]
        labels["sRSCU"] = [human_srscu_df[codon].values[0] if codon in human_srscu_df.columns else None for codon in labels["Codon"]]
        # replace '*' with 'Stop'
        labels["Amino Acid"] = ["Stop" if aa == "*" else aa
                                    for seq in data["protein"][:examples]
                                    for aa in seq[:max_length_data]
                                ]
    elif level == "sequence":
        # Sequence-level labels
        labels["Species"] = data["species_name"][:examples]
        labels["Gene"] = data["gene_name"][:examples]
        labels["GC content"] = [calculate_gc_content(seq[start:end])
                                for seq, start, end in zip(data["mrna"][:examples], data["codon_start"][:examples], data["codon_end"][:examples])]
        labels["sRSCU"] = [calculate_average_srscu(seq[start:end], "Homo sapiens", srscu_df)
                                for seq, start, end in zip(data["mrna"][:examples], data["codon_start"][:examples], data["codon_end"][:examples])]

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
        ax = sns.heatmap(attention, cmap='Purples', cbar=False) #, norm=LogNorm())
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
                           xlim=None, ylim=None):
    # Create a joint plot with scatter kind and custom settings
    g = sns.jointplot(x=x, y=y, data=df, kind="hex",
                      height=2.8, ratio=4, space=0.1, 
                      color="mediumpurple", alpha=0.5,  # mediumpurple, g, 0.7
                      marginal_kws=dict(bins=30, fill=True)
                      )
    # Check if highlighting is needed
    if highlighted_genes is not None:
        # Create a column to highlight genes
        df['highlight'] = df['record_id'].isin(highlighted_genes)
        colors = df['highlight'].map({True: 'mediumorchid', False: 'rebeccapurple'})
    else:
        colors = 'rebeccapurple' # mediumpurple, g, 0.7

    g.ax_joint.scatter(df[x], df[y], s=2, alpha=0.1, c=colors, edgecolor=None)

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
