import argparse
import json
import os
import pickle as pkl
import pprint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt

from utils import utils, model_utils
from learning.metrics import confusion_matrix_analysis
from learning.miou import IoU
from learning.weight_init import weights_init


class Train:
    def __init__(self, config):
        self.config = config


    def iterate(
    self, model, data_loader, criterion, optimizer=None, mode="train", device=None
    ):
        loss_meter = tnt.meter.AverageValueMeter()
        iou_meter = IoU(
            num_classes=self.config.num_classes,
            ignore_index=self.config.ignore_index,
            cm_device=self.config.device,
        )

        t_start = time.time()
        for i, batch in enumerate(data_loader):
            if device is not None:
                batch = self.recursive_todevice(batch, device)
            (x, dates), y = batch
            y = y.long()

            if mode != "train":
                with torch.no_grad():
                    out = model(x, batch_positions=dates)
            else:
                optimizer.zero_grad()
                out = model(x, batch_positions=dates)

            loss = criterion(out, y)
            if mode == "train":
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred = out.argmax(dim=1)
            iou_meter.add(pred, y)
            loss_meter.add(loss.item())

            if (i + 1) % self.config.display_step == 0:
                miou, acc = iou_meter.get_miou_acc()
                print(
                    "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                        i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                    )
                )

        t_end = time.time()
        total_time = t_end - t_start
        print("Epoch time : {:.1f}s".format(total_time))
        miou, acc = iou_meter.get_miou_acc()
        metrics = {
            f"{mode}_accuracy": acc,
            f"{mode}_loss": loss_meter.value()[0],
            f"{mode}_IoU": miou,
            f"{mode}_epoch_time": total_time,
        }

        return (metrics, iou_meter.conf_metric.value()) if mode == "test" else metrics

    def recursive_todevice(self, device):
        if isinstance(self, torch.Tensor):
            return self.to(device)
        elif isinstance(self, dict):
            return {k: self.recursive_todevice(v, device) for k, v in x.items()}
        else:
            return [self.recursive_todevice(c, device) for c in x]

    def prepare_output(self):
        os.makedirs(self.res_dir, exist_ok=True)
        for fold in range(1, 6):
            os.makedirs(os.path.join(self.res_dir, f"Fold_{fold}"), exist_ok=True)

    def checkpoint(self, log):
        with open(os.path.join(self.config.res_dir, f"Fold_{self}", "trainlog.json"), "w") as outfile:
            json.dump(log, outfile, indent=4)

    def save_results(self, metrics, conf_mat):
        with open(os.path.join(self.config.res_dir, f"Fold_{self}", "test_metrics.json"), "w") as outfile:
            json.dump(metrics, outfile, indent=4)
        pkl.dump(
            conf_mat,
            open(
                os.path.join(self.config.res_dir, f"Fold_{self}", "conf_mat.pkl"), "wb"
            ),
        )

    def overall_performance(self):
        cm = np.zeros((self.num_classes, self.num_classes))
        for fold in range(1, 6):
            cm += pkl.load(
                open(
                    os.path.join(self.res_dir, f"Fold_{fold}", "conf_mat.pkl"),
                    "rb",
                )
            )

        if self.ignore_index is not None:
            cm = np.delete(cm, self.ignore_index, axis=0)
            cm = np.delete(cm, self.ignore_index, axis=1)

        _, perf = confusion_matrix_analysis(cm)

        print("Overall performance:")
        print(f'Acc: {perf["Accuracy"]},  IoU: {perf["MACRO_IoU"]}')

        with open(os.path.join(self.res_dir, "overall.json"), "w") as file:
            file.write(json.dumps(perf, indent=4))
