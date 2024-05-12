import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
from torch import no_grad
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from misc import Network, mape_loss


class Trainer:
    def __init__(self, model: Network, train_loader: DataLoader, validation_loader: DataLoader,
                 lr: float, epochs: int, output_path: Path):
        self.__lr: float = lr
        self.__epochs: int = epochs
        self._criterion = mape_loss
        self.__model: Network = model
        self.__train_loader: DataLoader = train_loader
        self.__validation_loader: DataLoader = validation_loader

        self.__optimizer: Adam = Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=self.__lr, betas=(0.9, 0.99), eps=1e-15)

        self.__scaler: GradScaler = GradScaler(enabled=True)
        self.__lr_scheduler: StepLR = StepLR(self.__optimizer, step_size=10, gamma=0.1)
        self.__device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.__device)
        self.__ema: ExponentialMovingAverage = ExponentialMovingAverage(self.__model.parameters(), decay=0.95)

        self.__epochs: int = epochs
        self.__stats: Dict[str, Any] = {
            "loss": [],
            "valid_loss": [],
            "results": [], 
            "checkpoints": [],
            "best_result": None,
        }

        self.__output_path: Path = output_path
        self.__best_path: Path = self.__output_path / "best.pth"
        self.__checkpoints_path: Path = self.__output_path / "checkpoints"

        os.makedirs(self.__output_path, exist_ok=True)
        os.makedirs(self.__checkpoints_path, exist_ok=True)

        self._max_checkpoints: int = 2


    def __train_step(self, data):
        return self._criterion(self.__model(data["points"][0]), data["sdfs"][0])

    def __eval_step(self, data):
        return self.__train_step(data)


    def train(self, description: str = ""):    
        with tqdm(range(self.__epochs)) as pbar:
            current_status = {}

            def set_pbar_status(**kwargs):
                current_status.update(kwargs)
                pbar.set_description(f"Training{' ' if description else ''}{description} - " +
                                     ', '.join(f"{key}={val}" for key, val in current_status.items()))

            def on_train_loss_changed(new_loss: float):
                set_pbar_status(train_loss=f"{new_loss:.4f}")

            for epoch in pbar:
                self.__epochs = epoch

                self.__train_one_epoch(on_train_loss_changed)
                self.__save_checkpoint(full=True, best=False)
                self.__evaluate()
                set_pbar_status(val_loss=f"{self.__stats['valid_loss'][-1]:.4f}")
                pbar.refresh()
                self.__save_checkpoint(full=False, best=True)

    def __prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.__device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.__device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.__device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.__device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.__device, non_blocking=True)
        else: 
            data = data.to(self.__device, non_blocking=True)

        return data

    def __train_one_epoch(self, on_loss_changed: Optional[Callable[[float], None]] = None):
        total_loss = 0

        self.__model.train()

        step = 0

        for data in self.__train_loader:
            step += 1

            data = self.__prepare_data(data)

            self.__optimizer.zero_grad()

            with autocast(enabled=True):
                loss = self.__train_step(data)

            self.__scaler.scale(loss).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()

            self.__ema.update()

            loss_val = loss.item()
            total_loss += loss_val

            on_loss_changed(loss_val)

        average_loss = total_loss / step
        self.__stats["loss"].append(average_loss)

        self.__lr_scheduler.step()

    def __evaluate(self):
        total_loss = 0

        self.__model.eval()

        with no_grad():
            step = 0
            for data in self.__validation_loader:
                step += 1

                data = self.__prepare_data(data)

                self.__ema.store()
                self.__ema.copy_to()

                with autocast(enabled=True):
                    loss = self.__eval_step(data)

                self.__ema.restore()

                loss_val = loss.item()
                total_loss += loss_val

        average_loss = total_loss / step
        self.__stats["valid_loss"].append(average_loss)
        self.__stats["results"].append(average_loss)

    def __save_checkpoint(self, full=False, best=False):
        state = {
            'epoch': self.__stats,
            'stats': self.__stats,
        }

        if full:
            state['optimizer'] = self.__optimizer.state_dict()
            state['lr_scheduler'] = self.__lr_scheduler.state_dict()
            state['scaler'] = self.__scaler.state_dict()
            state['ema'] = self.__ema.state_dict()

        if not best:
            state['model'] = self.__model.state_dict()

            file_path = f"{self.__checkpoints_path}/epoch_{self.__epochs:03d}.pth"

            self.__stats["checkpoints"].append(file_path)

            if len(self.__stats["checkpoints"]) > self._max_checkpoints:
                old_ckpt = self.__stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.__stats["results"]) > 0:
                if self.__stats["best_result"] is None or self.__stats["results"][-1] < self.__stats["best_result"]:
                    self.__stats["best_result"] = self.__stats["results"][-1]

                    self.__ema.store()
                    self.__ema.copy_to()

                    state['model'] = self.__model.state_dict()

                    self.__ema.restore()

                    torch.save(state, self.__best_path)
