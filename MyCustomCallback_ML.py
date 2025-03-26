import json
import torch
from lightning import Callback
import os

class MyCustomCheckpoint(Callback):
    def __init__(self, save_dir, idx, epochs_of_interest, logger):
        super().__init__()
        self.save_dir = save_dir
        self.idx = idx
        self.epoch_set = set(epochs_of_interest)
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.epoch_set:
            metrics = trainer.callback_metrics

            res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in metrics.items()}
            self.logger.info(f"Client {self.idx} Local Model {epoch} epoch training result: {json.dumps(res_dict, indent=None)}")

            # filename = f"./{self.save_dir}/model-epoch={epoch}.ckpt"
            # trainer.save_checkpoint(filename)
            
            state_dict = pl_module.state_dict()
            
            os.makedirs(self.save_dir, exist_ok=True)
            filename = f"model-epoch={epoch}.pth"
            path = os.path.join(self.save_dir, filename)

            # Save the model state dictionary
            torch.save(state_dict, path)