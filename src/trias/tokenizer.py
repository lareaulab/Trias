import json
import os
import re
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer, AddedToken


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "target_vocab_file": "target_vocab.json"}

class TriasTokenizer(PreTrainedTokenizer):
    r"""

    This tokenizer tokenizes protein and codon sequences. The protein sequences have species names prepended to it.

    Args:

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (`int`, *optional*, defaults to 512):
            The maximum sentence length the model accepts.
	"""

    species_code_re = re.compile(">>(.*?)<<") 
    
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        target_vocab=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=2048,
        separate_vocabs=False,
        **kwargs,
    ) -> None:
        
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.separate_vocabs = separate_vocabs

        self.input_encoder = load_json(vocab_file)
        if separate_vocabs:
            self.output_encoder = load_json(target_vocab)
        else: 
            self.output_encoder = load_json(vocab_file)

        for encoder in [self.input_encoder, self.output_encoder]:
            if str(unk_token) not in encoder or str(pad_token) not in encoder:
                raise KeyError(f"Both {unk_token} and {pad_token} must be in the vocab")

        self.input_decoder = {v: k for k, v in self.input_encoder.items()}
        self.output_decoder = {v: k for k, v in self.output_encoder.items()}
        
        self.current_tokenizer = "protein_tokenizer"

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            target_vocab=target_vocab,
            separate_vocabs=separate_vocabs,
            **kwargs,
        )
    

    def _convert_token_to_id(self, token):
        if self.current_tokenizer == "protein_tokenizer":
            return self.input_encoder.get(token, self.input_encoder[self.unk_token])
        else:
            return self.output_encoder.get(token, self.output_encoder[self.unk_token])


    def remove_species_code(self, text: str):
        """Remove species codes like >>Homo sapiens<<"""
        match = self.species_code_re.match(text)
        code: list = [match.group(1)] if match else []
        return code, self.species_code_re.sub("", text).strip()


    def _tokenize(self, text: str) -> List[str]:
        if self.current_tokenizer == "protein_tokenizer":
            species_code, text = self.remove_species_code(text)
            pieces = list(text.lstrip())
            return species_code + pieces
        else:
            pieces = [text[i:i+3] for i in range(0, len(text), 3)]
            return pieces


    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the decoder."""
        if self.current_tokenizer == "protein_tokenizer":
            return self.input_decoder.get(index, self.unk_token)
        else:
            return self.output_decoder.get(index, self.unk_token)


    def batch_decode(self, sequences, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return super().batch_decode(sequences, **kwargs)


    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return super().decode(token_ids, **kwargs)


    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]


    def __call__(self, text, text_target=None, add_special_tokens=True, padding=False, truncation=False, max_length=None, return_tensors=None, **kwargs):
        # Convert single strings to lists of strings if necessary
        if isinstance(text, str):
            text = [text]
        
        if text_target is not None and isinstance(text_target, str):
            text_target = [text_target]

        # Tokenize input sequences (protein sequences)
        self._switch_to_input_mode()
        input_encodings = [self._tokenize(t) for t in text]
        input_ids = [self.convert_tokens_to_ids(tokens) for tokens in input_encodings]

        # Tokenize target sequences (codon sequences) if provided
        if text_target is not None:
            self._switch_to_output_mode()
            target_encodings = [self._tokenize(tgt) for tgt in text_target]
            target_ids = [self.convert_tokens_to_ids(tokens) for tokens in target_encodings]
        else:
            target_ids = None

        # Calculate the number of special tokens to be added
        num_special_tokens = 1 if add_special_tokens else 0  # Adjust based on how many special tokens you add

        # Truncate sequences if truncation is enabled and max_length is provided
        if truncation and max_length is not None:
            truncation_length = max_length - num_special_tokens
            input_ids = [ids[:truncation_length] for ids in input_ids]  # Truncate to fit special tokens
            if target_ids is not None:
                target_ids = [ids[:truncation_length] for ids in target_ids]



        # Add special tokens (such as <eos>) if requested
        if add_special_tokens:
            input_ids = [self.build_inputs_with_special_tokens(ids) for ids in input_ids]
            if target_ids is not None:
                target_ids = [self.build_inputs_with_special_tokens(ids) for ids in target_ids]

        # Pad sequences if padding is required
        if padding:
            input_ids = self.pad(
                {"input_ids": input_ids},
                padding=True,
                max_length=max_length,
                return_tensors=None 
            )["input_ids"]

            if target_ids is not None:
                target_ids = self.pad(
                    {"input_ids": target_ids},
                    padding=True,
                    max_length=max_length,
                    return_tensors=None
                )["input_ids"]

        # Create attention masks (1 for actual tokens, 0 for padding)
        attention_mask = [[1 if token != self.pad_token_id else 0 for token in ids] for ids in input_ids]

        # Prepare the final output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Include labels if they exist (for seq2seq tasks)
        if target_ids is not None:
            output["labels"] = target_ids

        # Convert to tensors if requested (e.g., PyTorch)
        if return_tensors:
            for key, value in output.items():
                output[key] = torch.tensor(value)

        return output


    def _switch_to_input_mode(self):
        self.current_tokenizer = "protein_tokenizer"

    def _switch_to_output_mode(self):
        self.current_tokenizer = "codon_tokenizer"


    @property
    def vocab_size(self) -> int:
        if self.current_tokenizer == "protein_tokenizer":
            return len(self.input_encoder)
        else:
            return len(self.output_encoder)


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            print(f"Vocabulary path ({save_directory}) should be a directory")
            return
        saved_files = []

        out_input_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
        )
        out_output_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["target_vocab"]
        )

        save_json(self.input_encoder, out_input_vocab_file)
        if self.separate_vocabs:
            save_json(self.output_encoder, out_output_vocab_file)
        saved_files.extend([out_input_vocab_file, out_output_vocab_file])

        return tuple(saved_files)

    
    def get_vocab(self) -> Dict:
        if self.current_tokenizer == "protein_tokenizer":
            return self.get_src_vocab()
        else:
            return self.get_tgt_vocab()


    def get_src_vocab(self):
        return dict(self.input_encoder, **self.added_tokens_encoder)


    def get_tgt_vocab(self):
        return dict(self.output_encoder, **self.added_tokens_decoder)


    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)
    

