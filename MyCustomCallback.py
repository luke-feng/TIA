import json
import os
from lightning import Callback
import torch

class MyCustomCheckpoint(Callback):
    def __init__(self, save_dir, idx, rou=None, logger=None):
        super().__init__()
        self.save_dir = save_dir
        self.idx = idx
        self.rou = rou 
        self.logger = logger
        
        #self.target_round = 9
        
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        if self.rou in [9, 14, 19, 24, 29, 34, 39]:
            metrics = trainer.callback_metrics
    
            res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in metrics.items()}
            
            if self.logger is not None:
              self.logger.info(f"Client {self.idx} Local Model training result: {json.dumps(res_dict, indent=None)}")
            
            #if self.rou == self.target_round:
            state_dict = pl_module.state_dict()
            filename = f"client_{self.idx}.pth"
            path = os.path.join(self.save_dir, filename)
    
            # Save the model state dictionary
            torch.save(state_dict, path)
            
        
    '''def on_train_start(self, trainer, pl_module):

        state_dict = pl_module.state_dict()
        filename = f"pre_client_{self.idx}.pth"
        path = os.path.join(self.save_dir, filename)

        # Save the model state dictionary
        torch.save(state_dict, path)'''