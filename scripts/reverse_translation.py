import torch
import argparse
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.utils import logging as hf_logging
from trias import *

import warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--protein_sequence", type=str, required=True, help="Amino acid sequence ending with *")
    parser.add_argument("--species", type=str, required=True, help="Species name")
    parser.add_argument(
        "--decoding",
        type=str,
        choices=["greedy", "beam", "nucleus"],
        default="greedy",
        help="greedy: deterministic. beam: deterministic, explores beam_width paths. "
             "nucleus: stochastic, samples from top-p mass (different output each run unless --seed is set).",
    )
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling cumulative-probability cutoff.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to make nucleus sampling reproducible.")
    return parser.parse_args()

def run_inference(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = BartForConditionalGeneration.from_pretrained(args.model_path).to(device)

    input_seq = f">>{args.species}<< {args.protein_sequence}"
    input_ids = tokenizer.encode(input_seq, return_tensors="pt").to(device)

    if args.decoding == "greedy":
        outputs = model.generate(input_ids, max_length=tokenizer.model_max_length)
    elif args.decoding == "beam":
        outputs = model.generate(
            input_ids,
            num_beams=args.beam_width,
            early_stopping=True,
            max_length=tokenizer.model_max_length,
        )
    else:  # nucleus
        outputs = model.generate(
            input_ids,
            do_sample=True,
            top_p=args.top_p,
            max_length=tokenizer.model_max_length,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated codon sequence:\n{result}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
