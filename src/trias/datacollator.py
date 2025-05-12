import torch
from transformers import DataCollatorForSeq2Seq


def mask_tokens(inputs, tokenizer, mlm_probability=0.15, num_prefix_tokens=1):
    """ Prepare masked tokens inputs for masked language modeling """
    probability_matrix = torch.full(inputs.shape, mlm_probability)
    
    # Ensure that the first `num_prefix_tokens` tokens are not masked
    special_tokens_mask = [
        [1] * num_prefix_tokens + tokenizer.get_special_tokens_mask(val[num_prefix_tokens:], already_has_special_tokens=True) 
        for val in inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    inputs[masked_indices] = tokenizer.mask_token_id

    return inputs


class DataCollatorForBART(DataCollatorForSeq2Seq):    
    def __init__(self, tokenizer, model, mlm_probability=0.15, num_prefix_tokens=1):
        super().__init__(tokenizer, model)
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        self.num_prefix_tokens = num_prefix_tokens

    def __call__(self, features):
        batch = super().__call__(features)
        batch['input_ids'] = mask_tokens(batch['input_ids'], self.tokenizer, self.mlm_probability, self.num_prefix_tokens)
        return batch
