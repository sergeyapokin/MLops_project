import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig
from hydra import compose, initialize


class AirModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size, 
            hidden_size=cfg.hidden_size, 
            num_layers=cfg.num_layers, 
            batch_first=cfg.batch_first
        )
        self.linear = nn.Linear(cfg.hidden_size, cfg.output_size)
        self.save_hyperparameters(logger=False)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
    
    def training_step(self, batch):
        loss_fn = nn.MSELoss()
        X_batch, y_batch = batch
        out, _ = self.lstm(X_batch)
        y_pred = self.linear(out)
        loss = loss_fn(y_pred, y_batch)
        self.log("train loss", loss)
        return loss

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
    def save_model(self, checkpoint_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.trainer.optimizers[0].state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizers[0].load_state_dict(checkpoint['optimizer_state_dict'])


def get_config():
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="train_config")
    return cfg


def train(X_train, y_train, path_to_save_model):
    cfg = get_config()
    pl_air_model = AirModel(cfg.lstm)
    logger = pl.loggers.MLFlowLogger(
        cfg.experiment.name,
        tracking_uri='http://127.0.0.1:5000'
    )
    trainer = pl.Trainer(
        accelerator="cpu", 
        max_epochs=cfg.data.num_epochs, 
        logger=logger
    )
    loader = DataLoader(
        TensorDataset(X_train, y_train), 
        shuffle=cfg.data.shuffle,
        batch_size=cfg.data.batch_size
    )
    
    trainer.fit(pl_air_model, loader)
    trainer.save_checkpoint(path_to_save_model)
    # mlflow_logging()
    return pl_air_model
