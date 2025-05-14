from transformers import AutoConfig, AutoTokenizer
from .tokenizer import TriasTokenizer
from .configuration import TriasConfig


AutoConfig.register("trias", TriasConfig)
AutoTokenizer.register(TriasConfig, TriasTokenizer)
