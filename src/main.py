import gc
import os

import torch

if os.path.exists("/kaggle"):
    PATH_DATASET = "/kaggle/working/PROJETO_PESS_DADOS"
    os.chdir("/kaggle/working/PROJETO_PESS_DADOS/src")
    os.environ["AMBIENTE"] = "KAGGLE"
elif os.path.exists("/content"):
    os.environ["AMBIENTE"] = "COLAB"
else:
    PATH_DATASET = os.path.abspath(".")
    os.environ["AMBIENTE"] = "LOCAL"

import mlflow
from DataProcesser.data import StrokeDataset
from Models.mlp import MLP
from lightning import Callback, seed_everything, Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.pytorch import autolog
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from Models.optimizer import Optimizer



## -----------------------------COLAR NO KAGGLE------------------
def main():
    ###------SEEDS---------###
    RAND_SEED = 42
    seed_everything(RAND_SEED)
    ## ----------VARIAVEIS TREINO-----------
    BATCH_SIZE = 8
    cpus = os.cpu_count()
    WORKERS = cpus if cpus is not None else 1
    EPOCHS = 2
    TRIALS = 50
    #### -------- VARIAVEIS DE LOGGING ------------
    EXP_NAME = "stroke_1"
    RUN_NAME: str | None = None  # noma da RUN: pode ser aleatório ou definido
    MLF_TRACK_URI = "sqlite:///mlflow.db"
    AMBIENTE = os.environ["AMBIENTE"]

    mlflow.set_tracking_uri(MLF_TRACK_URI)
    mlflow.set_experiment(EXP_NAME)
    autolog(log_models=True, checkpoint=True, exclusive=False)

    ## ----------VARIAVEIS MODELO-----------
    HIDN_DIMS = 32
    N_CLASSES = 2
    N_LAYERS = 5

    dataset = StrokeDataset()
    train_loader, val_loader = dataset.create_dataloaders(BATCH_SIZE, WORKERS)

    INPUT_DIMS = dataset.data.shape[1]
    model = MLP(INPUT_DIMS, HIDN_DIMS, N_LAYERS, N_CLASSES)
    # loop principal de treinamento
    def objective(trial:optuna.Trial):
        # Suggest hyperparameters
        hidn_dims = trial.suggest_int("hidn_dims", 16, 128, step=16)
        n_layers = trial.suggest_int("n_layers", 2, 6)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        
        # Recreate dataloaders with trial batch_size
        train_loader, val_loader = dataset.create_dataloaders(batch_size, WORKERS)
        
        # Create model with trial hyperparameters
        model = MLP(INPUT_DIMS, hidn_dims, n_layers, N_CLASSES, learning_rate=learning_rate)
        
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
            active_run_id = run.info.run_id

            mlflow_logger = MLFlowLogger(
                experiment_name=EXP_NAME,
                tracking_uri=MLF_TRACK_URI,
                log_model=True,
                run_id=active_run_id,
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            )
            
            trainer = Trainer(
                max_epochs=EPOCHS,
                devices=1,
                accelerator="gpu" if AMBIENTE == "KAGGLE" else "cpu",
                logger=mlflow_logger,
                enable_checkpointing=False,
                callbacks=[early_stopping],
            )

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            mlflow.log_params(trial.params)
            
            val_loss = trainer.callback_metrics["val_loss"].item()
            torch.cuda.empty_cache()
            gc.collect()
            
            return val_loss

    with mlflow.start_run(run_name=RUN_NAME) as parent_run:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=TRIALS)
        
        # Log best parameters
        mlflow.log_params({"best_" + k: v for k, v in study.best_trial.params.items()}, run_id=parent_run.info.run_id)
        mlflow.log_metric("best_val_loss", study.best_trial.value or 0, run_id=parent_run.info.run_id)
        
        print("Best hyperparameters:", study.best_trial.params)
        print("Best validation loss:", study.best_trial.value)
        # torch.cuda.empty_cache()
        # return trainer.callback_metrics["val_loss"].item()


    # study = optuna.create_study(direction="minimize")
    # study.optimize(optimize, n_trials=TRIALS, timeout=60*10)
    # print("Best hyperparameters:", study.best_trial.params)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
    if os.environ["AMBIENTE"] == "LOCAL":
        from visualyze import see_model

        see_model()
