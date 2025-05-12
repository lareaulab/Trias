from transformers import Seq2SeqTrainer, Trainer, get_cosine_schedule_with_warmup
import torch


class CustomTrainer(Seq2SeqTrainer):
    """
    Trainer class for training models with custom settings.

    Args:
        train_len (int): Length of the training dataset.
    """

    def __init__(self, train_len, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.max_steps > 0:
            self.max_epochs = (self.args.per_device_train_batch_size * self.args.max_steps) // train_len + 1
            self.max_steps = self.args.max_steps
        else:
            self.max_epochs = self.args.num_train_epochs
            self.max_steps = (self.max_epochs * train_len) // self.args.per_device_train_batch_size + 1
    
        self.t_initial = int((train_len / self.args.per_device_train_batch_size) * self.max_epochs)
        self.warmup_t = int(0.01 * self.t_initial) 

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.args.learning_rate,
                                            weight_decay=self.args.weight_decay)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer = self.optimizer,
                                                            num_warmup_steps = self.warmup_t,
                                                            num_training_steps = self.max_steps)
        
