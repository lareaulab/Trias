from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DatasetArguments:
    """
    Arguments used to specifiy dataset features.
    """

    dataset_name: str = field(metadata={"help": "Name or path of the input dataset."})
    train_len: int = field(metadata={"help": "Length of the training dataset, required for learning rate scheduler."})
    max_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximal length of the input sequences."}
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Vocabulary dictionary for protein input sequence."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The cache directory where the dataset will be stored."}
    )


@dataclass
class ModelArguments:
    """
    Arguments used to specifiy model features.
    """
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Name or path of the model weights. Use only to fine-tune pre-trained model."}
    )
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of tokens in the input sequences. Used for positional embeddings."}
    )
    encoder_layers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of layers in the encoder."}
    )
    encoder_ffn_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The dimensionality of the feed-forward network in the encoder."}
    )
    encoder_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "The number of attention heads in the encoder."}
    )
    decoder_layers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of layers in the decoder."}
    )
    decoder_ffn_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The dimensionality of the feed-forward network in the decoder."}
    )
    decoder_attention_heads: Optional[int] = field(
        default=None,
        metadata={"help": "The number of attention heads in the decoder."}
    )
    d_model: Optional[int] = field(
        default=None,
        metadata={"help": "The dimensionality of the model's hidden states."}
    )
