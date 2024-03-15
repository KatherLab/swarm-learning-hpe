import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation

import logging
import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.ncsnv3 import NCSNv3Deepest
from losses import get_optimizer
from datasets import (get_dataset, data_transform,
                      inverse_data_transform, Gamma_correction)

from models import anneal_Langevin_dynamics
from models import get_sigmas
from models.ema import EMAHelper
from swarmlearning.pyt import SwarmCallback

__all__ = ['NCSNRunner']


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):

        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = NCSNv3Deepest(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        swarmCallback = SwarmCallback(
            totalEpochs=10,
            syncFrequency=256,
            minPeers=1,
            maxPeers=1,
            nodeWeightage=100,
            model=score,
            #mergeMethod="geomedian",
        )
        swarmCallback.on_train_begin()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            # load it
            states = torch.load(os.path.join(
                self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
            score.load_state_dict(states[0], strict=True)
            # Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]-1
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

            print(
                f'Pretrained model from epoch {start_epoch} and step {step} successfully loaded')

        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            # Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        n_epochs = self.config.training.n_epochs

        for epoch in range(start_epoch+1, n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                loss = anneal_dsm_score_estimation(scorenet=score,
                                                   sigmas=sigmas,
                                                   samples=X,
                                                   anneal_power=self.config.training.anneal_power,
                                                   hook=hook)

                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_hook()

                # logging.info("step: {}, loss: {}, average loss:{}".format(
                #     step, loss.item(), avg_loss / num_items))

                logging.info(
                    f"epoch: {epoch}/{n_epochs}, step: {step}, loss: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                swarmCallback.on_batch_end()

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()

                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(scorenet=test_score,
                                                                    sigmas=sigmas,
                                                                    samples=test_X,
                                                                    anneal_power=self.config.training.anneal_power,
                                                                    hook=hook)
                        tb_logger.add_scalar(
                            'test_loss', test_dsm_loss, global_step=step)
                        test_tb_hook()
                        # logging.info("step: {}, test_loss: {}".format(
                        #     step, test_dsm_loss.item()))

                        logging.info(
                            f"step: {step}, test_loss: {test_dsm_loss.item():.4f}")

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(
                        self.args.log_path, 'checkpoint_{}.pt'.format(step)))
                    torch.save(states, os.path.join(
                        self.args.log_path, 'checkpoint.pt'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        # Different part from NeurIPS 2019.
                        # Random state will be affected because of sampling during training time.
                        init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(
                            self.config, init_samples)

                        all_samples = anneal_Langevin_dynamics(scorenet=test_score,
                                                               sigmas=sigmas.cpu().numpy(),
                                                               x_mod=init_samples,
                                                               n_steps_each=self.config.sampling.n_steps_each,
                                                               step_lr=self.config.sampling.step_lr,
                                                               final_only=True,
                                                               verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        save_image(sample,
                                   os.path.join(
                                       self.args.log_sample_path, 'image_grid_{}.png'.format(step)),
                                   nrow=int(sample.shape[0] ** 0.5))

                        torch.save(sample, os.path.join(
                            self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                        del test_score
                        del all_samples
        swarmCallback.on_train_end()


    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(
                self.args.log_path, 'checkpoint.pt'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pt'),
                                map_location=self.config.device)

        print(f'Pre-trained model loaded')

        score = NCSNv3Deepest(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)
        print(score.parameters)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        score.eval()

        print(f'Now sampling {self.config.sampling.batch_size} images ...')

        for idx in range(self.config.sampling.num_batches):

            init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                      self.config.data.image_size, self.config.data.image_size,
                                      device=self.config.device)

            init_samples = data_transform(self.config, init_samples)

            all_samples = anneal_Langevin_dynamics(scorenet=score,
                                                   sigmas=sigmas,
                                                   x_mod=init_samples,
                                                   n_steps_each=self.config.sampling.n_steps_each,
                                                   step_lr=self.config.sampling.step_lr,
                                                   final_only=True,
                                                   verbose=True,
                                                   denoise=self.config.sampling.denoise)

            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size)
            sample = inverse_data_transform(self.config, sample)

            save_image(sample,
                       os.path.join(self.args.image_folder,
                                    f'image_gen_{idx}.jpg'),
                       nrow=int(sample.shape[0] ** 0.5))
