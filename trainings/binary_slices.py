from pathlib import Path
from sklearn.model_selection import train_test_split
from DiagnosisAI.datasets_torch.brain_slices_dataset import BrainSlicesDataset
from DiagnosisAI.lightining_modules.binary_slices_segment import BinarySegmenterUnet
import torch as t
import pytorch_lightning as pl
import json
import os


# ======== PARAMS ========
pl_model = BinarySegmenterUnet()
path = "../datasets/brain/train_images_max_area/"
checkpoint_path = "./checkpoints/binary_segment_unet/resnet34/"
model_state_path = "./model_states/binary_segment_unet/resnet34/"
batch_size = 8
num_workers = 4
train_size = 0.7
seed = 42

# ========CHECK DIR EXISTS ==========
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(model_state_path):
    os.makedirs(model_state_path)

# ==========================

def __get_filenames_from_dir__(path:str):
    filenames_pickle = []
    for dir in Path(path).iterdir():
        for filename in dir.iterdir():
            rel_path = Path(*filename.parts[-2:])
            filenames_pickle.append(rel_path)

    return filenames_pickle

# ======== SPLIT FILENAMES ==========
print("Getting filenames ...")
test_size = 1 - train_size
filenames = __get_filenames_from_dir__(path)
train_names, val_names = train_test_split(filenames, test_size=test_size, random_state=seed)
val_names, test_names = train_test_split(val_names, test_size=0.5, random_state=seed)

# ======== LOAD DATA ===========
print("Loading datasets")
train_dataset = BrainSlicesDataset(train_names)
val_dataset = BrainSlicesDataset(val_names)
test_dataset = BrainSlicesDataset(test_names)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

img, label = next(iter(train_loader))
print("Shapes:")
print(f"Input img: {img.shape},  Label: {label.shape}")

# ========== MODEL ================
f = open('../config/secret.json')
api_key = json.load(f)

model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=25)
logger = pl.loggers.NeptuneLogger(
    api_key=api_key['api_neptune'],
    project="jumpincrane/Binary-slices-brain-seg-net"
)

print("Training model ...")
trainer = pl.Trainer(logger=logger, callbacks=[model_checkpoint, early_stopping], gpus=1, max_epochs=100)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

logger.run.stop()

print("Saving model state ...")
t.save(pl_model.state_dict(), model_state_path)