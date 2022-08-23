import torch as t
from torch.utils import data
import torch.nn as nn
from DiagnosisAI.models.resnet3d import generate_model
from DiagnosisAI.utils.metrics import calc_metrics, calculate_type_errors
import os
import time
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from DiagnosisAI.datasets_torch.brain_3d_dataset import Brain3DDataset
from DiagnosisAI.models.Unet3D import Unet3d
from DiagnosisAI.models.Unet3D_fromgit import UNet


# ======== PARAMS ========
checkpoint_path = "./checkpoints/seg_binary_3d/unet3d_fromgit_t1ce/"
model_state_path = "./model_states/seg_binary_3d/unet3d_fromgit_t1ce/"
log_path = "./logs/seg_binary_3d/unet3d_fromgit_t1ce/"
dataset_root_path = "../datasets/brain/Brats2021_training_df/"
batch_size = 2
num_workers = 4
train_size = 0.7
seed = 42
device = t.device('cuda')

# ========CHECK DIR EXISTS ==========
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(model_state_path):
    os.makedirs(model_state_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

# ===================================
def __get_filenames_from_dir__(path:str):
    filenames = []
    for dir in Path(path).iterdir():
        filenames.append(dir)
    return filenames

test_size = 1 - train_size
filenames = __get_filenames_from_dir__(dataset_root_path)
train_names, val_names = train_test_split(filenames, test_size=test_size, random_state=seed)
val_names, test_names = train_test_split(val_names, test_size=0.5, random_state=seed)

# ======== LOAD DATA ===========
print("Loading datasets")
train_dataset = Brain3DDataset(train_names, t1ce=True)
val_dataset = Brain3DDataset(val_names, t1ce=True)
test_dataset = Brain3DDataset(test_names, t1ce=True)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

img, label = next(iter(train_loader))
print("Shapes:")
print(f"Input img: {img.shape},  Label: {label.shape}")


# ========== MODEL ================
network = UNet(num_channels=1, out_channels=1, residual='pool').to(device)
# ======= SET OPTIMIZER LOSS FUNC ========
optimizer = t.optim.AdamW(network.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()

train_losses_epoch = []
val_losses_epoch = []
acc_epoch = []
recall_epoch = []
f1_score_epoch = []
precision_epoch = []

# ==== TRAINING AND VALID ========
start_time = time.time()
for epoch in range(60):  # loop over the dataset multiple times
    network.train()
    train_loss = 0.0
    valid_loss = 0.0
    epoch_tp = 0
    epoch_fp = 0
    epoch_tn = 0
    epoch_fn = 0

    for i, batch in enumerate(tqdm(train_loader), 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = network(inputs)

        loss = criterion(outputs, labels.to(dtype=t.float32, device=device))
        loss.backward()
        optimizer.step()

        tp, fp, tn, fn = calculate_type_errors(outputs, labels, mode='segment_binary')
        
        # print statistics
        epoch_tp += tp
        epoch_fp += fp
        epoch_tn += tn
        epoch_fn += fn
        train_loss += loss.item()

    print("Saving model state per epoch ...")
    t.save(network.state_dict(), model_state_path + f"model_state")
    t.save(optimizer.state_dict(), checkpoint_path + f"optimizer_adam")

    network.eval()
    for j, batch in enumerate(val_loader, 0):    
        inputs, labels = batch
        inputs = inputs.to(device)
        inputs = inputs.to(device)
        with t.no_grad():    
            outputs = network(inputs)
        loss = criterion(outputs, labels.to(dtype=t.float32, device=device))
        # Calculate Loss
        valid_loss += loss.item()

    # calc metrics
    recall, precision, acc, f1_score = calc_metrics(epoch_tp.sum(), epoch_fp.sum(), epoch_tn.sum(), epoch_fn.sum())

    print(f"[Epoch:{epoch}], Training loss:{train_loss / (i + 1)}, Val loss:{valid_loss / (j + 1)}")

    train_losses_epoch.append(train_loss / (i + 1))
    val_losses_epoch.append(valid_loss / (j + 1))
    recall_epoch.append(recall.item())
    precision_epoch.append(precision.item())
    f1_score_epoch.append(f1_score.item())
    acc_epoch.append(acc.item())

end_time = time.time()
print(f'Finished Training: {(end_time - start_time) / 60} minutes')

print("Saving model state ...")
t.save(network.state_dict(), model_state_path + "model_state")
print("Saving logs ...")
df = pd.DataFrame({'train_loss': train_losses_epoch,
                    'val_loss': val_losses_epoch,
                    'recall': recall_epoch,
                    'precision': precision_epoch,
                    'accuracy': acc_epoch,
                    'f1_score': f1_score_epoch
                })
df.to_csv(log_path + "logs.csv")
