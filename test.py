import torch 
torch.set_float32_matmul_precision("medium")

from Dataset.mnist import MNISTDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from Model.mlp import MLP
import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset

from Dataset.svhn import SVHNdataset
from Model.svhnresnet18 import SVHNResNet18
from Model.pcamresnet18 import PCAMResNet18

from Dataset.pcam import PCAMdataset
import time
from Dataset.imagenet100 import ImageNet100Dataset
from Model.imagenet100vgg import ImageNet100VGG
from Model.imagenet100st import ImageNet100SwinTransformer
from Model.imagenet100deit import ImageNet100DeiTTiny
from Model.imagenet100mobilevit import ImageNet100MobileViT
from Model.imagenet100poolformer import ImageNet100PoolFormerS12
from Model.pcamdesnet import PCAMDenseNet121
from Model.pcamefficientnetv2 import PCAMEfficientNetB0
from Model.pcamShuffleNetV2 import PCAMShuffleNetV2

dataset = ImageNet100Dataset(img_size=128)

train_indices = np.arange(len(dataset.train_set))
np.random.shuffle(train_indices)

# data_size=500
num_client=10
data_size = min(int(round(len(train_indices)/num_client/2)) - 1, 3600)
train_subsets_indices = [train_indices[i * data_size:(i + 1) * data_size] for i in range(num_client)]
train_subsets = [Subset(dataset.train_set, indices) for indices in train_subsets_indices]
# train_loader = DataLoader(train_subsets[9], batch_size=128, shuffle=True, num_workers=0)
# for (x,y) in train_loader:
#     print(y)
#     break

start = time.time()
for i in range(1):   

    train_loader = DataLoader(train_subsets[i], batch_size=128, shuffle=True, num_workers=0)
    model = ImageNet100PoolFormerS12()

    local_trainer = Trainer(max_epochs=3, accelerator="auto", devices="auto", logger=False,
                                                            # callbacks = [MyCustomCheckpoint(save_dir=f"{model_directory}/Local_models/Round_{r}",
                                                            # idx=client.idx, rou=r, logger=model_logger)],
                                                            enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=True)
                                            
    local_trainer.fit(model, train_loader)  

end = time.time()
print(f"trainig time {end-start} s")  
test_indices = np.arange(len(dataset.test_set))
np.random.shuffle(test_indices)
data_size=500
num_client=10
test_subsets_indices = [test_indices[i * data_size:(i + 1) * data_size] for i in range(num_client)]
test_subsets = [Subset(dataset.test_set, indices) for indices in test_subsets_indices]
test_loader = DataLoader(test_subsets[0], batch_size=32, shuffle=False, num_workers=0)
local_trainer.test(model, test_loader)
