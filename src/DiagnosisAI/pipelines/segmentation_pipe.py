from abc import abstractmethod, ABC
import json
from typing import Any, Union, Optional
import os
from datetime import datetime
import time
from pathlib import Path


from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, precision, recall, f1_score, jaccard_index
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, LRScheduler

from diagnosisai.models.segmentation.unet import UNet
from diagnosisai.utils.training import DefaultEarlyStopper


LOSS_FUNCTIONS = {"CrossEntropyLoss": nn.CrossEntropyLoss,
                  "BCEWithLogistLoss": nn.BCEWithLogitsLoss}

OPTIMIZERS = {"AdamW": AdamW,
              "SGD": SGD}

SCHEDULERS = {"Exponential": ExponentialLR,
              "Linear": LinearLR}

METRIC_DICT = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1score": f1_score,
               "jaccard_index": jaccard_index}


class BasicSegmentationPipeline(ABC):
    """ Base class for classification pipelines """

    def __init__(self, config_path: str, config_name: str) -> None:
        self.model_params, self.train_params = self._load(config_path, config_name)

    @abstractmethod
    def fit(self):
        """
        Method in which model training is performed and the result should be the best model.
        """
        pass

    @abstractmethod
    def predict(self) -> torch.Tensor:
        """  """
        pass

    def _load(self, config_path: str, config_name: str) -> tuple[dict[str, Any]]:
        """ Load model, training parameters """
        with open(config_path) as f:
            params = json.load(f)
            try:
                params = params[config_name]
            except KeyError:
                raise Exception(f"Configuration named {config_name} not found in configuration file")

        return params['model_params'], params['train_params']

    def _select_optimizer(self, optimizer_name: str, optimizer_params: dict[str, Any]) -> Optimizer:
        try:
            optimizer = OPTIMIZERS[optimizer_name](self.model.parameters(), **optimizer_params)
        except KeyError:
            raise NotImplementedError(f"The demanded optimizer {optimizer_name} is not implemented.")

        return optimizer

    def _select_scheduler(self, scheduler_name: str, scheduler_params: dict[str, Any]) -> LRScheduler:
        try:
            scheduler = SCHEDULERS[scheduler_name](self.optim, **scheduler_params)
        except KeyError:
            raise NotImplementedError(f"The demanded scheduler {scheduler_name} is not implemented.")

        return scheduler

    def _select_loss_func(self, loss_func_name: str, loss_func_params: dict[str, Any]) -> nn.Module:
        for param_name, param_value in loss_func_params.items():
            if isinstance(param_value, list):
                loss_func_params[param_name] = torch.tensor(param_value)
        try:
            loss_func = LOSS_FUNCTIONS[loss_func_name](**loss_func_params)
        except KeyError:
            raise NotImplementedError(f"The demanded loss function {loss_func_name} is not implemented.")

        return loss_func


class SegmentationPipeline(BasicSegmentationPipeline):
    """
    Segmentation Pipeline based on a specified configuration.

    :param Optional[str] config_path: Path to the configuration file.
    :param Optional[str] config_name: Name of the configuration (default is "UNet").
    :param Optional[str] model_path: Path to the pre-trained model file (default is None).
    """

    def __init__(self, config_path: Optional[str] = None, config_name: Optional[str] = "UNet", model_path: Optional[str] = None) -> None:

        if model_path:
            self.fitted = True
            with open(model_path, 'rb') as f:
                self.model = torch.load(f)
        else:
            super().__init__(config_path, config_name)

            self.fitted = False

            self.model = UNet(**self.model_params)

            self.loss_func = self._select_loss_func(loss_func_name=self.train_params['loss_func'],
                                                    loss_func_params=self.train_params['loss_params'])
            self.optim = self._select_optimizer(optimizer_name=self.train_params['optimizer'],
                                                optimizer_params=self.train_params['optimizer_params'])
            self.scheduler = self._select_scheduler(scheduler_name=self.train_params['scheduler'],
                                                    scheduler_params=self.train_params['scheduler_params'])
            self.early_stopper = DefaultEarlyStopper(patience=self.train_params["epoch_patience"])

    def fit(self, train_data: Dataset, val_data: Dataset):
        """
        Fit the model to the training data.

        :param Dataset train_data: Training data.
        :param Dataset val_data: Validation data.
        """
        if self.fitted:
            raise RuntimeError("Model is already loaded from given path. If you want to train/fit model don't pass model path")

        try:
            self._training_loop(train_data, val_data, self.train_params, self.model,
                                self.loss_func, self.optim, self.early_stopper, self.scheduler)
        except KeyboardInterrupt:
            self.fitted = True
        else:
            self.fitted = True

    def predict(self, test_data: Dataset, batch_size: int = 16) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Make predictions on the test data.

        :param Dataset test_data: Test data.
        """

        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        accecelator = Accelerator()

        model, test_dl = accecelator.prepare(self.model, test_dl)
        model = model.eval()

        all_gts = []
        all_preds = []
        running_metrics = {}

        for batch in tqdm(test_dl):

            inputs, labels = batch
            with torch.no_grad():
                y_pred = model.forward(inputs)

            all_gts.append(labels.cpu())
            all_preds.append(y_pred.cpu())

        all_preds = torch.concatenate(all_preds)

        if all_preds.shape[1] == 1:
            all_preds = torch.sigmoid(all_preds)
            metrics_mode = "binary"
        else:
            all_preds = torch.softmax(all_preds, dim=1)
            metrics_mode = "multiclass"

        for metric_name, metric_func in METRIC_DICT.items():
            running_metrics.setdefault(f"{metric_name}", 0.0)
            running_metrics[f"{metric_name}"] = metric_func(
                all_preds,
                torch.concatenate(all_gts),
                task=metrics_mode, num_classes=model.out_channels).item()

        return all_preds, running_metrics

    def _training_loop(
            self, train_data: Dataset, val_data: Dataset, train_params: dict[str, Any],
            model: UNet, loss_func: nn.Module, optimizer, early_stopper, scheduler):

        train_dl = DataLoader(train_data, train_params['batch_size'], shuffle=True,
                              num_workers=train_params['num_workers'])
        val_dl = DataLoader(val_data, train_params['batch_size'], shuffle=False,
                            num_workers=train_params['num_workers'])

        accecelator = Accelerator()

        model, optimizer, scheduler, train_dl, val_dl, loss_func = accecelator.prepare(
            model, optimizer, scheduler, train_dl, val_dl, loss_func)

        dls = {"train": train_dl, "val": val_dl}

        if not os.path.exists(train_params['save_folder_path']):
            os.mkdir(train_params['save_folder_path'])

        current_run = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

        writer = SummaryWriter(Path(train_params['save_folder_path'], "logs", current_run))

        t_tqdm = tqdm(range(train_params['epochs']))
        best_model = model
        min_val_loss = torch.inf

        for epoch in t_tqdm:
            running_metrics = {}
            for phase, data_l in dls.items():
                phase_gt = []
                phase_pred = []

                if phase == "train":
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(data_l):
                    if i % 100 == 0:
                        print(f"Step {i}/{len(data_l)}")

                    inputs, labels = batch
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred, loss = model.step(inputs, labels, loss_func)
                        if phase == "train":
                            accecelator.backward(loss)
                            optimizer.step()

                    phase_gt.append(labels.cpu())
                    phase_pred.append(y_pred.detach().cpu())

                    running_metrics.setdefault(f"{phase}_loss", 0.0)
                    running_metrics[f"{phase}_loss"] += loss.item()

                all_preds = torch.concatenate(phase_pred)

                if all_preds.shape[1] == 1:
                    all_preds = torch.sigmoid(all_preds)
                    metrics_mode = "binary"
                else:
                    all_preds = torch.softmax(all_preds, dim=1)
                    metrics_mode = "multiclass"

                for metric_name, metric_func in METRIC_DICT.items():
                    running_metrics.setdefault(f"{phase}_{metric_name}", 0.0)
                    running_metrics[f"{phase}_{metric_name}"] = metric_func(
                        all_preds,
                        torch.concatenate(phase_gt),
                        task=metrics_mode, num_classes=model.out_channels)

            for metric_name in running_metrics:
                if 'train' in metric_name:
                    if 'loss' in metric_name:
                        writer.add_scalar(metric_name, running_metrics[metric_name] / len(train_dl), epoch)
                    else:
                        writer.add_scalar(metric_name, running_metrics[metric_name], epoch)
                elif 'val' in metric_name:
                    if 'loss' in metric_name:
                        writer.add_scalar(metric_name, running_metrics[metric_name] / len(val_dl), epoch)
                    else:
                        writer.add_scalar(metric_name, running_metrics[metric_name], epoch)

            scheduler.step()

            val_loss = running_metrics.get("val_loss") / len(val_data)

            if val_loss < min_val_loss:
                best_model = model
                min_val_loss = val_loss

                with open(os.path.join(train_params['save_folder_path'],
                                       "logs",
                                       current_run,
                                       "best_model.pt"), "wb") as f:
                    torch.save(best_model, f)

            if early_stopper.early_stop(val_loss):
                break

            if epoch % train_params['save_freq_epoch'] == 0:
                with open(os.path.join(train_params['save_folder_path'],
                                       "logs",
                                       current_run,
                                       f"model_{current_run}_{epoch}.pt"), "wb") as f:
                    torch.save(model, f)

        writer.flush()
        writer.close()
