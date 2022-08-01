from cProfile import run
import medmnist
import torch as t
from torch.utils import data
import torch.nn as nn
from DiagnosisAI.models.resnet3d import generate_model
from medmnist import INFO
from DiagnosisAI.utils.metrics import calc_metrics, calculate_type_errors
import os
import time
import pandas as pd


# ======== PARAMS ========
checkpoint_path = "./checkpoints/classif_3d/resnet18/"
model_state_path = "./model_states/classif_3d/resnet18/"
log_path = "./logs/classif_3d/resnet18/"
batch_size = 8
num_workers = 8
train_size = 0.7
seed = 42
data_flag = 'organmnist3d'
download = True

# ========CHECK DIR EXISTS ==========
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(model_state_path):
    os.makedirs(model_state_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

# ==========================

info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train',  download=download)
val_dataset = DataClass(split='val',  download=download)
test_dataset = DataClass(split='test',  download=download)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

img, label = next(iter(train_loader))
print("Shapes:")
print(f"Input img: {img.shape},  Label: {label.shape}")

# ========== MODEL ================
network = generate_model(model_depth=18, n_classes=11, n_input_channels=1)
# ======= SET OPTIMIZER LOSS FUNC ========
optimizer = t.optim.AdamW(network.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

train_losses_epoch = []
val_losses_epoch = []
acc_epoch = []
recall_epoch = []
f1_score_epoch = []
precision_epoch = []

# ==== TRAINING AND VALID ========
start_time = time.time()
for epoch in range(1000):  # loop over the dataset multiple times
    train_loss = 0.0
    valid_loss = 0.0
    network.train()
    epoch_tp = 0
    epoch_fp = 0
    epoch_tn = 0
    epoch_fn = 0

    for i, batch in enumerate(train_loader, 0):
        inputs, labels = batch
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = network(inputs.type(t.float32))
        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=11)
        labels = labels.squeeze()

        loss = criterion(outputs, labels.type(t.float32))
        loss.backward()
        optimizer.step()

        pred_classes = t.argmax(outputs, dim=1)
        target_classes = t.argmax(labels, dim=1)
        tp, fp, tn, fn = calculate_type_errors(pred_classes, target_classes, mode='classif_multiclass', num_classes=11)
        
        # print statistics
        epoch_tp += tp
        epoch_fp += fp
        epoch_tn += tn
        epoch_fn += fn
        train_loss += loss.item()

    network.eval()
    for j, (inputs, labels) in enumerate(val_loader, 0):        
        outputs = network(inputs.type(t.float32))
        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=11)
        labels = labels.squeeze()
        loss = criterion(outputs, labels.type(t.float32))
        # Calculate Loss
        valid_loss += loss.item()

    # calc metrics
    recall, precision, acc, f1_score = calc_metrics(epoch_tp.sum(), epoch_fp.sum(), epoch_tn.sum(), epoch_fn.sum())

    print(f"[Epoch:{epoch}], Training loss:{train_loss / (i + 1)}, Val loss:{valid_loss / (j + 1)}")

    train_losses_epoch.append(train_loss / (i + 1))
    val_losses_epoch.append(valid_loss / (j + 1))
    recall_epoch.append(recall)
    precision_epoch.append(precision_epoch)
    f1_score_epoch.append(f1_score)
    acc_epoch.append(acc)

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
