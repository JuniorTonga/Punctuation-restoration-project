from copy import deepcopy
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import time


class Wrapper:
    """
    Wrapper classs that takes a model, loss function and optimizer and gives an
    easy to use interface to train and evaluate the model, in addition to
    plotting loss w.r.t. iterations

    Attributes
    ----------
    model : Model
        the model to be trained
    loss_fn : object
        the loss function used during training
    optimizer : object
        the optimizer used to update weights
    device : str
        the device used to train the model can be either 'cpu' or 'cuda'
    train_loader : object
    val_loader : object
    losses : list
    val_losses : list
    total_epochs : int
    best_params : dict

    Methodes
    --------
    __init__ :
    to :
    set_loaders :
    _train_step :
    _val_step :
    _mini_batch :
    set_seed :
    train :
    predict :
    plot_losses :
    save_checkpoint :
    load_checkpoint :
    make_lr_fn :
    lr_range_test :
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        use_attention: bool = False,
    ) -> None:
        """
        Initialize the model, loss_fn, optimizer and the device. Then move the
        model to the choosen device

        Parameters
        ----------
        self : Wrapper
        model : Model
        loss_fn : object
        optimizer : object

        Returns
        -------
        None

        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.use_attention = use_attention

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.losses = []
        self.val_losses = []
        self.scores = {}
        self.val_scores = {}
        self.execution_times = []
        self.val_execution_times = []
        self.total_epochs = 0

        self.best_params = model.state_dict()
        self.best_val_loss = np.inf

    def set_seed(self, seed: int = 42) -> None:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            pass

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        np.random.seed(seed)

        try:
            self.train_loader.sample.generator.manual_seed(seed)
        except AttributeError:
            pass

    def to(self, device: str) -> None:
        """
        Specify different device

        Parameters
        ----------
        device : str

        Returns
        -------
        None
        """
        self.device = device
        self.model.to(self.device)

    def set_loaders(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Set dataloaders for train and/or validation

        Parameters
        ----------
        train_loader : object
        val_loader : object

        Returns
        -------
        None
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _train_step(self, x: Any, y: Any, *args, **kwargs) -> Any:
        self.model.train()

        yhat = self.model(x, *args)

        loss = self.loss_fn(yhat.permute(0, 2, 1), y)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return yhat, loss.item()

    def _val_step(self, x: Any, y: Any, *args, **kwargs) -> Any:
        self.model.eval()

        yhat = self.model(x, *args)

        loss = self.loss_fn(yhat.permute(0, 2, 1), y)

        return yhat, loss.item()

    def _mini_batch(
        self, validation: bool = False, metrics: Any = [], return_y_hat: bool = False
    ) -> Optional[Any]:
        if validation:
            data_loader = self.val_loader
            step = self._val_step
        else:
            data_loader = self.train_loader
            step = self._train_step

        if data_loader is None:
            return None

        ys = []
        y_hats = []
        y_masks = []
        n_batches = len(data_loader)
        n_metrics = len(metrics)
        mini_batch_losses = np.zeros((n_batches, 1))
        mini_batch_metrics = np.zeros((n_batches, n_metrics))

        start_time = time.time()
        for i, (x_batch, y_batch, attn_mask_batch, y_mask_batch) in enumerate(
            data_loader
        ):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            attn_mask_batch = attn_mask_batch.to(self.device)
            y_mask_batch = y_mask_batch.to(self.device)

            if self.use_attention:
                y_hat, mini_batch_loss = step(x_batch, y_batch, attn_mask_batch)
            else:
                y_hat, mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses[i][0] = mini_batch_loss

            ys.append(y_batch)
            y_hats.append(y_hat)
            y_masks.append(y_mask_batch)
            if n_metrics > 0:
                for j, metric in enumerate(metrics):
                    mini_batch_metrics[i][j] = torch.as_tensor(
                        metric["fn"](
                            y_batch.detach().cpu().numpy(),
                            y_hat.detach().cpu().numpy(),
                            y_mask_batch.detach().cpu().numpy(),
                        )
                    )
        end_time = time.time()
        exec_time = end_time - start_time

        loss = np.mean(mini_batch_losses)
        if n_metrics > 0:
            scores = np.mean(mini_batch_metrics, axis=0)
        else:
            scores = None

        if return_y_hat:
            return ys, y_hats, y_masks, loss, scores, exec_time
        else:
            return loss, scores, exec_time

    def train(self, n_epochs: int, metrics: Any = [], seed: int = 42) -> None:
        """
        Parameters
        ----------
        metrics : list of dict
            every element has two fields, the name "name" of the metric and the
            metric funciton "fn"
        """
        self.set_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1
            mini_batch_output = self._mini_batch(validation=False, metrics=metrics)
            if mini_batch_output is None:
                return None
            loss, scores, exec_time = mini_batch_output
            self.losses.append(loss)
            for i, metric in enumerate(metrics):
                if not metric["name"] in self.scores:
                    self.scores[metric["name"]] = []
                self.scores[metric["name"]].append(scores[i])
            self.execution_times.append(exec_time)

            with torch.no_grad():
                mini_batch_output = self._mini_batch(validation=True, metrics=metrics)
                if mini_batch_output is None:
                    return None
                val_loss, val_scores, val_exec_time = mini_batch_output
                self.val_losses.append(val_loss)
                for i, metric in enumerate(metrics):
                    if not metric["name"] in self.val_scores:
                        self.val_scores[metric["name"]] = []
                    self.val_scores[metric["name"]].append(val_scores[i])
                self.val_execution_times.append(val_exec_time)
                if val_loss < self.best_val_loss:
                    self.best_params = deepcopy(self.model.state_dict())

            if len(metrics) == 0:
                max_metric_name_len = 20
            else:
                max_metric_name_len = max(
                    20, max([len(metric["name"]) for metric in metrics])
                )

            header = ("{:" + str(max_metric_name_len) + "d} | {:20} | {:20}").format(
                epoch, "train scores", "validation scores"
            )

            print("-" * len(header))
            print(header)
            print("-" * len(header))

            print(
                ("{:" + str(max_metric_name_len) + "} | {:20.10f} | {:20.10f}").format(
                    "loss", self.losses[-1], self.val_losses[-1]
                )
            )
            # print("-" * len(header))

            for metric, score, val_score in zip(metrics, scores, val_scores):
                print(
                    (
                        "{:" + str(max_metric_name_len) + "} | {:20.10f} | {:20.10f}"
                    ).format(metric["name"], score, val_score)
                )
                # print("-" * len(header))

            print(
                ("{:" + str(max_metric_name_len) + "} | {:20.10f} | {:20.10f}").format(
                    "execution time (s)",
                    self.execution_times[-1],
                    self.val_execution_times[-1],
                )
            )
            print("-" * len(header))

    def report(self, metric):
        self.restore_best_params()

        with torch.no_grad():
            mini_batch_output = self._mini_batch(
                validation=True, metrics=[], return_y_hat=True
            )
            if mini_batch_output is None:
                return None
            y, y_hat, y_mask, val_loss, val_scores, val_exec_time = mini_batch_output

            y = torch.cat(y).detach().cpu().numpy()
            y_hat = torch.cat(y_hat).detach().cpu().numpy()
            y_mask = torch.cat(y_mask).detach().cpu().numpy()

        precision_per_class = (
            metric(
                y, y_hat, y_mask, score="precision", average=None, labels=[0, 1, 2, 3]
            )
            * 100
        )
        precision_overall = (
            metric(y, y_hat, y_mask, score="precision", average="macro") * 100
        )

        recall_per_class = (
            metric(y, y_hat, y_mask, score="recall", average=None, labels=[0, 1, 2, 3])
            * 100
        )
        recall_overall = metric(y, y_hat, y_mask, score="recall", average="macro") * 100

        f1_per_class = (
            metric(y, y_hat, y_mask, score="f1", average=None, labels=[0, 1, 2, 3])
            * 100
        )
        f1_overall = metric(y, y_hat, y_mask, score="f1", average="macro") * 100

        score_per_class = zip(precision_per_class, recall_per_class, f1_per_class)

        print("\n===== scores report =====")
        print(
            (" {:15} | {:15} | {:15} | {:15} | {:15}").format(
                "[EMPTY]", "[PERIOD]", "[COMMA]", "[QUESTION]", "OVERALL"
            )
        )
        print("-" * 89)

        print((("  P  |  R  |  F  |" * 4) + "  P  |  R  |  F  "))
        print("-" * 89)

        for score in score_per_class:
            print(
                ("{:5.2f}|{:5.2f}|{:5.2f}|").format(score[0], score[1], score[2]),
                end="",
            )
        print(
            ("{:5.2f}|{:5.2f}|{:5.2f}|").format(
                precision_overall, recall_overall, f1_overall
            )
        )
        print("-" * 89)

    def score(self, metric):
        self.restore_best_params()

        with torch.no_grad():
            mini_batch_output = self._mini_batch(
                validation=True, metrics=[], return_y_hat=True
            )
            if mini_batch_output is None:
                return None
            y, y_hat, y_mask, val_loss, val_scores, val_exec_time = mini_batch_output

            y = torch.cat(y).detach().cpu().numpy()
            y_hat = torch.cat(y_hat).detach().cpu().numpy()
            y_mask = torch.cat(y_mask).detach().cpu().numpy()

        f1_overall = metric(y, y_hat, y_mask, score="f1", average="macro")
        return f1_overall

    def restore_best_params(self) -> None:
        self.model.load_state_dict(self.best_params)

    def predict_val(self):
        self.restore_best_params()

        with torch.no_grad():
            mini_batch_output = self._mini_batch(
                validation=True, metrics=[], return_y_hat=True
            )
            if mini_batch_output is None:
                return None
            y, y_hat, y_mask, val_loss, val_scores, val_exec_time = mini_batch_output

            y = torch.cat(y).detach().cpu().numpy()
            y_hat = torch.cat(y_hat).detach().cpu().numpy().argmax(-1)
            y_mask = torch.cat(y_mask).detach().cpu().numpy()
        return y, y_hat

    def predict(self, x: Tensor) -> np.ndarray:
        self.model.eval()

        x_tensor = torch.as_tensor(x).float()

        y_hat_tensor = self.model(x_tensor.to(self.device))

        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self) -> plt.Figure:
        fig = plt.figure(figsize=(20, 7))
        ax1, ax2 = fig.subplots(1, 2)
        ax1.semilogy(self.losses, label="Training Loss", c="b")
        if self.val_loader:
            ax1.semilogy(self.val_losses, label="Validation Loss", c="r")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()

        for metric in self.scores:
            print(metric)
            ax2.plot(self.scores[metric], label="Training Score " + metric)
            if self.val_loader:
                ax2.plot(self.val_scores[metric], label="Validation score " + metric)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Score")
        ax2.legend()

        return fig

    def save_checkpoint(self, filename: str) -> None:
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "val_loss": self.val_losses,
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]

        self.model.train()

    @staticmethod
    def make_lr_fn(start_lr: float, end_lr: float, n_iter: int, step_mode="exp") -> Any:
        if step_mode == "linear":
            factor = (end_lr / start_lr - 1) / n_iter

            def lr_fn(iteration):
                return 1 + iteration * factor

        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / n_iter

            def lr_fn(iteration):
                return np.exp(factor) ** iteration

        return lr_fn

    def lr_range_test(
        self,
        data_loader: DataLoader,
        end_lr: float,
        n_iter: int = 100,
        step_mode="exp",
        alpha: float = 0.05,
        ax: Optional[plt.Axes] = None,
    ) -> Any:
        previous_states = {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }

        start_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

        lr_fn = Wrapper.make_lr_fn(start_lr, end_lr, n_iter)
        scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_fn)

        tracking = {"loss": [], "lr": []}
        iteration = 0

        while iteration < n_iter:
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_hat = self.model(x_batch)

                loss = self.loss_fn(y_hat, y_batch)
                loss.backward()

                tracking["lr"].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking["loss"].append(loss.item())
                else:
                    prev_loss = tracking["loss"][-1]
                    smooth_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking["loss"].append(smooth_loss)

                iteration += 1

                if iteration == n_iter:
                    break

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        self.optimizer.load_state_dict(previous_states["optimizer"])
        self.model.load_state_dict(previous_states["model"])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()

        ax.plot(tracking["lr"], tracking["loss"])
        if step_mode == "exp":
            ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        return tracking, fig
