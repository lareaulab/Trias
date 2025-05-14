from transformers import BartConfig, AutoTokenizer
from .tokenizer import TriasTokenizer

AutoTokenizer.register(BartConfig, TriasTokenizer)
