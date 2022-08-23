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
from tqdm import tqdm

# ======== PARAMS ========
checkpoint_path = "./checkpoints/classif_3d_new/resnet18/"
model_state_path = "./model_states/classif_3d_new/resnet18/"
log_path = "./logs/classif_3d_new/resnet18/"
batch_size = 8
num_workers = 8
train_size = 0.7
seed = 42
data_flag = 'organmnist3d'
download = True
device = t.device('cuda')

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
network = generate_model(model_depth=18, n_classes=11, n_input_channels=1).to(device)
# ======= SET OPTIMIZER LOSS FUNC ========
optimizer = t.optim.AdamW(network.parameters(), lr=3e-5)
criterion = nn.CrossEntropyLoss()

train_losses_epoch = []
val_losses_epoch = []
acc_epoch = []
recall_epoch = []
f1_score_epoch = []
precision_epoch = []
acc_val_epoch = []
recall_val_epoch = []
f1_score_val_epoch = []
precision_val_epoch = []

# ==== TRAINING AND VALID ========
start_time = time.time()
for epoch in range(400):  # loop over the dataset multiple times
    train_loss = 0.0
    valid_loss = 0.0
    network.train()
    epoch_tp = 0
    epoch_fp = 0
    epoch_tn = 0
    epoch_fn = 0
    epoch_val_tp = 0
    epoch_val_fp = 0
    epoch_val_tn = 0
    epoch_val_fn = 0

    for i, batch in enumerate(tqdm(train_loader), 0):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = network(inputs.type(t.float32))
        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=11)
        labels = labels.squeeze()

        try:
            loss = criterion(outputs, labels.to(dtype=t.float32, device=device))
        except:
            outputs.shape
            labels = labels.unsqueeze(0)
            loss = criterion(outputs, labels.to(dtype=t.float32, device=device))


        loss.backward()
        optimizer.step()

        pred_classes = t.argmax(outputs.cpu(), dim=1)
        target_classes = t.argmax(labels.cpu(), dim=1)
        tp, fp, tn, fn = calculate_type_errors(pred_classes, target_classes, mode='classif_multiclass', num_classes=11)
        
        # print statistics
        epoch_tp += tp.numpy()
        epoch_fp += fp.numpy()
        epoch_tn += tn.numpy()
        epoch_fn += fn.numpy()
        train_loss += loss.item()

    print("Saving model state ...")
    t.save(network.state_dict(), model_state_path + "model_state")

    network.eval()
    for j, (inputs, labels) in enumerate(val_loader, 0): 
        inputs = inputs.to(device)
        labels = labels.to(device)  

        outputs = network(inputs.type(t.float32))
        outputs = nn.Softmax(dim=1)(outputs)
        labels = nn.functional.one_hot(labels, num_classes=11)
        labels = labels.squeeze()

        try:
            loss = criterion(outputs, labels.to(dtype=t.float32, device=device))
        except:
            outputs.shape
            labels = labels.unsqueeze(0)
            loss = criterion(outputs, labels.to(dtype=t.float32, device=device))

        pred_classes = t.argmax(outputs.cpu(), dim=1)
        target_classes = t.argmax(labels.cpu(), dim=1)
        tp, fp, tn, fn = calculate_type_errors(pred_classes, target_classes, mode='classif_multiclass', num_classes=11)

        # Calculate Loss
        epoch_val_tp += tp.numpy()
        epoch_val_fp += fp.numpy()
        epoch_val_tn += tn.numpy()
        epoch_val_fn += fn.numpy()
        valid_loss += loss.item()

    # calc metrics
    recall, precision, acc, f1_score = calc_metrics(epoch_tp.sum(), epoch_fp.sum(), epoch_tn.sum(), epoch_fn.sum())
    val_recall, val_precision, val_acc, val_f1_score = calc_metrics(epoch_val_tp.sum(), epoch_val_fp.sum(), epoch_val_tn.sum(), epoch_val_fn.sum())

    print(f"[Epoch:{epoch}], Training loss:{train_loss / (i + 1)}, Val loss:{valid_loss / (j + 1)}")

    train_losses_epoch.append(train_loss / (i + 1))
    val_losses_epoch.append(valid_loss / (j + 1))
    recall_epoch.append(recall)
    precision_epoch.append(precision)
    f1_score_epoch.append(f1_score)
    acc_epoch.append(acc)
    recall_val_epoch.append(val_recall)
    precision_val_epoch.append(val_precision)
    f1_score_val_epoch.append(val_f1_score)
    acc_val_epoch.append(val_acc)


    # print("Saving logs ...")
    df = pd.DataFrame({'train_loss': train_losses_epoch,
                        'val_loss': val_losses_epoch,
                        'recall': recall_epoch,
                        'precision': precision_epoch,
                        'accuracy': acc_epoch,
                        'f1_score': f1_score_epoch,
                        'val_recall': recall_val_epoch,
                        'val_precision': precision_val_epoch,
                        'val_accuracy': acc_val_epoch,
                        'val_f1_score': f1_score_val_epoch
                    })
    df.to_csv(log_path + "logs.csv")


end_time = time.time()
print(f'Finished Training: {(end_time - start_time) / 60} minutes')



