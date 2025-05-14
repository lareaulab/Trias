from transformers import AutoTokenizer
from .tokenizer import TriasTokenizer
from .configuration import BartConfig

AutoTokenizer.register(BartConfig, TriasTokenizer, exist_ok=True))
