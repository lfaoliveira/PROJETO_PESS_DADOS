##codigo dos modelos
import numpy as np
import pandas as pd
from torch import optim
import torch.nn as nn
import torch
from Models.interface import ClassificationModel
from Models.utils import calc_metrics


class MLP(ClassificationModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        n_layers: int,
        num_classes: int,
        **kwargs,
    ):
        self.hyperparams: dict[str, float | int] = kwargs.get("hyperparameters", {})

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims, dtype=torch.float32),
            nn.ReLU(),
        )
        for _ in range(n_layers):
            self.model.append(nn.LazyLinear(hidden_dims, dtype=torch.float32))
            self.model.append(nn.SELU())

        self.model.append(nn.Linear(hidden_dims, num_classes, dtype=torch.float32))
        self.example_input_array = torch.zeros(input_dim, dtype=torch.float32)
        self.save_hyperparameters()
        # print(self.model)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # training_step defines the train loop.
        # it is independent of forward
        data, labels = batch
        # print("TREINAMENTO: ", data.shape, labels.shape)

        logits = self.model(data)
        labels = torch.squeeze(labels.long())

        loss = nn.functional.cross_entropy(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        # self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # training_step defines the train loop.
        # it is independent of forward

        data, labels = batch
        # print(data.shape, labels.shape)
        logits = self.model(data)
        labels = torch.squeeze(labels.long())

        loss = nn.functional.cross_entropy(logits, labels)

        f_beta, prec, rec, roc_auc = calc_metrics(labels, logits)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_prec", float(prec), prog_bar=False)
        self.log("val_rec", float(rec), prog_bar=False)
        self.log("val_f_beta", float(f_beta), prog_bar=False)
        self.log("val_roc_auc", float(roc_auc), prog_bar=False)
        return loss

    def configure_optimizers(self):
        lr = self.hyperparams.get("lr", 1e-5)
        beta0 = self.hyperparams.get("beta0", 0.99)
        beta1 = self.hyperparams.get("beat1", 0.9999)
        weight_decay = self.hyperparams.get("weight_decay", 1e-5)
        optimizer = optim.Adam(
            self.parameters(), lr=lr, betas=(beta0, beta1), weight_decay=weight_decay
        )
        return optimizer

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        output_df: pd.DataFrame,
        test_dataset: torch.utils.data.Subset,
    ):
        data, labels = batch
        logits = self.model(data)
        labels = torch.squeeze(labels.long())
        predictions = torch.argmax(logits, dim=1)

        results = np.empty(predictions.shape[0], dtype=np.dtype("U4"))

        # Get dataset indices from the Subset's indices attribute
        start_idx = batch_idx * data.shape[0]
        end_idx = start_idx + data.shape[0]
        dataset_indices = test_dataset.indices[start_idx:end_idx]

        for i in range(predictions.shape[0]):
            if predictions[i] == 1 and labels[i] == 1:
                results[i] = "TP"
            elif predictions[i] == 1 and labels[i] == 0:
                results[i] = "FP"
            elif predictions[i] == 0 and labels[i] == 1:
                results[i] = "FN"
            else:
                results[i] = "TN"

        # Get original untransformed data
        original_data = [
            test_dataset.dataset.original_df.iloc[idx] for idx in dataset_indices
        ]

        batch_results = pd.DataFrame(
            {
                "dataset_indices": dataset_indices,
                "predictions": predictions.numpy(force=True),
                "labels": labels.numpy(force=True),
                "diagnostic": results,
                "original_data": original_data,
            }
        )
        output_df = pd.concat([output_df, batch_results], ignore_index=True)

        return output_df
