from DiagnosisAI.datasets_torch.chest_dataset import ChestDataset
from DiagnosisAI.lightining_modules.chest_model import ChestModel
from pathlib import Path
import torch as t
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

chest_path = Path("../datasets/chest_xray/")
checkpoint_path = Path('../trainings/model_states/')
normal_filenames = []
virus_filenames = []
bacteria_filenames = []

for dirs in chest_path.iterdir():
  dir_name = str(dirs)
  for filenames in dirs.iterdir():
    if "NORMAL" in dir_name:
      normal_filenames.append(filenames)
    if "VIRUS" in dir_name:
      virus_filenames.append(filenames)
    if "BACTERIA" in dir_name:
      bacteria_filenames.append(filenames)

# normal split
train_n_filenames, rest_n_filenames = train_test_split(normal_filenames, test_size=0.3, random_state=42)
valid_n_filenames, test_n_filenames = train_test_split(rest_n_filenames, test_size=0.5, random_state=42)

# virus split
train_v_filenames, rest_v_filenames = train_test_split(virus_filenames, test_size=0.3, random_state=42)
valid_v_filenames, test_v_filenames = train_test_split(rest_v_filenames, test_size=0.5, random_state=42)

# bacteria split
train_b_filenames, rest_b_filenames = train_test_split(bacteria_filenames, test_size=0.3, random_state=42)
valid_b_filenames, test_b_filenames = train_test_split(rest_b_filenames, test_size=0.5, random_state=42)

# 0 - normal
# 1 - bacteria
# 2 - virus

train_filenames = train_n_filenames + train_b_filenames + train_v_filenames
train_classes = np.full(len(train_n_filenames), 0).tolist() + np.full(len(train_b_filenames), 1).tolist() + np.full(len(train_v_filenames), 2).tolist()
valid_filenames = valid_n_filenames + valid_b_filenames + valid_v_filenames
valid_classes = np.full(len(valid_n_filenames), 0).tolist() + np.full(len(valid_b_filenames), 1).tolist() + np.full(len(valid_v_filenames), 2).tolist()
test_filenames = test_n_filenames + test_b_filenames + test_v_filenames
test_classes = np.full(len(test_n_filenames), 0).tolist() + np.full(len(test_b_filenames), 1).tolist() + np.full(len(test_v_filenames), 2).tolist()

train_dataset = ChestDataset(train_filenames, train_classes)
valid_dataset = ChestDataset(valid_filenames, valid_classes)
test_dataset = ChestDataset(test_filenames, test_classes)

batch_size = 8
train_chest_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_chest_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_chest_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# training
chest_model = ChestModel()
logger = pl.loggers.CSVLogger("logs", name="chest_xrays")
model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=30)
trainer = pl.Trainer(logger=logger, callbacks=[model_checkpoint, early_stopping], gpus=1, max_epochs=150)

trainer.fit(chest_model, train_dataloaders=train_chest_dataloader, val_dataloaders=val_chest_dataloader)
t.save(chest_model.network.state_dict(), "./checkpoint_model_state.ckpt")