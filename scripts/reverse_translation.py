import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--protein_sequence", type=str, required=True, help="Amino acid sequence ending with *")
    parser.add_argument("--decoding", type=str, choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_width", type=int, default=5)
    return parser.parse_args()

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    inputs = tokenizer(args.protein_sequence, return_tensors="pt").to(device)

    if args.decoding == "greedy":
        outputs = model.generate(**inputs)
    else:
        outputs = model.generate(**inputs, num_beams=args.beam_width, early_stopping=True)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated codon sequence:\n{result}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
