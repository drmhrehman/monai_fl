import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from monai.inferers import sliding_window_inference

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data_batch) in enumerate(self.data_loader):
            data, target = data_batch['img'].to(self.device), data_batch['seg'].to(self.device)

            # if data.isnan().sum() > 1 or data.isinf().sum():
            #     print('debug')
            # if target.isnan().sum() > 1 or target.isinf().sum():
            #     print('debug')
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            # save videos to tensorboard
            with torch.no_grad():
                # save images to tensorboard
                if batch_idx == 0:
                    self.writer.writer.add_video(tag='train/image', vid_tensor=np.moveaxis(data.cpu().add(1).mul(128).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)
                    self.writer.writer.add_video(tag='train/ground_truth', vid_tensor=np.moveaxis(target.cpu().mul(255).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)
                    self.writer.writer.add_video(tag='train/prediction', vid_tensor=np.moveaxis(output.cpu().mul(255).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data_batch) in enumerate(self.valid_data_loader):
                data, target = data_batch['img'].to(self.device), data_batch['seg'].to(self.device)
                output_logits = sliding_window_inference(data,
                                                  sw_batch_size=3,
                                                  roi_size=(128, 128, 128),
                                                  predictor=self.model,
                                                  overlap=0.25,
                                                  do_sigmoid=False)
                output = torch.sigmoid(output_logits)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                if batch_idx == 0:
                    self.writer.writer.add_video(tag='val/image', vid_tensor=np.moveaxis(data.cpu().add(1).mul(128).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)
                    self.writer.writer.add_video(tag='val/ground_truth', vid_tensor=np.moveaxis(target.cpu().mul(255).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)
                    self.writer.writer.add_video(tag='val/prediction', vid_tensor=np.moveaxis(output.cpu().mul(255).numpy().astype(np.uint8), 4, 1),
                                                 global_step=self.writer.step,
                                                 fps=12)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # save metrics averaged over the full validation set to tensorboard
        for met in self.metric_ftns:
            self.writer.writer.add_scalar(f'{met.__name__}/full_validation_set', self.valid_metrics.result()[met.__name__], epoch)
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
