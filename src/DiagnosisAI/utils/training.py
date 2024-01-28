import torch


class DefaultEarlyStopper:
    """It stops the training loop if the given requirements are met: for `patience` epochs the validation loss
    didn't improve by `min_delta` threshold.

    :param int patience: determines patience before stopping training (how many times it is to allow the criterion
        not to be met), defaults to `5`.
    :param int min_delta: determines potential margin.
    """

    def __init__(self, patience: int = 5, min_delta: int = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss: float) -> bool:
        """Check if requirements for early stop are met: the validation loss has stopped improving for `patience` epochs.

        :param float validation_loss: The current validation loss.
        :return bool: Value if stop training loop.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False