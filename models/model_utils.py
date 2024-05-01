# Import basic pkg
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# Import custom pkg
from models.making_model import get_model
from utils.smooth_croos_entropy import LabelSmoothingCrossEntropyLoss

class Classifier(pl.LightningModule):
    def __init__(self, args, TB_logger):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = get_model(args)
        self.TB_logger = TB_logger

        # Label Smoothing Croos Entropy Loss
        self.loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        self.conv_init = args.conv_init
        self.linear_init = args.linear_init
        self.linear_init_std_dev = args.linear_init_std_dev

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc_1 = (1.0 * (F.softmax(logits, 1).argmax(1) == y)).mean()
        acc_5 = (1.0 * (torch.topk(F.softmax(logits, 1), k=5, dim=1)[1] == y.view(-1, 1)).sum(dim=1).bool().float().mean())
        
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, prog_bar=True)
        
        # logger.info(acc)
        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc_1', acc_1, prog_bar=True)
        self.log(f'{part}_acc_5', acc_5, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.args.learning_rate, weight_decay = self.args.optimizer_weight_decay)
        
        # Define the learning rate scheduler
        total_iterations = self.args.epoch
        warmup_iterations = int(self.args.lr_max_time * total_iterations)
        linear_scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor = self.args.start_factor, 
                                                    end_factor=1.0, 
                                                    total_iters=warmup_iterations)
        
        # Define the cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=total_iterations - warmup_iterations, 
                                                                eta_min = (self.args.learning_rate) * self.args.start_factor)
        
        return [optimizer], [linear_scheduler, cosine_scheduler]


    # Initialize weight with kaiming and trunc
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            if self.linear_init == "trunc_normal":
                nn.init.trunc_normal_(module.weight, mean=0, std=self.linear_init_std_dev)
            elif self.linear_init == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            else:
                raise ValueError(f"Invalid initialization method: {self.linear_init}")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def training_epoch_end(self, outputs):
        # log histogram of parameters
        # Tensorboard is used
        for name, param in self.named_parameters():
            self.TB_logger.experiment.add_histogram(name, param, self.current_epoch)