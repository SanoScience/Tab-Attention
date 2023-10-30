"""Train script for fetal ultrasound data analysis."""
import pandas as pd
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from models.resnet_tabattention import ResNetTabAttention
import time

from models.resnet import r3d_18
from models.daft import DAFT, InteractiveHNN
from video_data_loader import FetalWeightVideo
import argparse
import warnings
from metrics import save_fold_metrics, save_metrics_all, save_per_patient_results

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="Multi-task learning for fetal US biometry.")
parser.add_argument("--data",
                    type=str,
                    default="../data/",
                    help="Path to the data directory.")
parser.add_argument("--x_img_size",
                    type=int,
                    default=128,
                    help="Input X image size.")
parser.add_argument("--y_img_size",
                    type=int,
                    default=128,
                    help="Input Y image size")
parser.add_argument("--num_workers",
                    type=int,
                    default=8,
                    help="Number of workers for processing the data.")
parser.add_argument("--n_classes",
                    type=int,
                    default=1,
                    help="Number of classes.")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="Number of batch size.")
parser.add_argument("--epochs",
                    type=int,
                    default=250,
                    help="Number of epochs.")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="Number of learning rate.")
parser.add_argument("--w_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay.")
parser.add_argument("--parallel",
                    type=bool,
                    default=True,
                    help="Parallel learning on GPU.")
parser.add_argument("--GPU",
                    type=bool,
                    default=True,
                    help="Use GPU.")
parser.add_argument("--display_steps",
                    type=int,
                    default=10,
                    help="Number of display steps.")
parser.add_argument("--model_name",
                    type=str,
                    default="TabAttention",
                    help="Name of trained model.")
parser.add_argument("--frames_num",
                    type=int,
                    default=16,
                    help="Number of frames in chunk")
parser.add_argument("--skip_frames",
                    type=int,
                    default=2,
                    help="Number of frames to skip")
parser.add_argument("--pixels_crop",
                    type=int,
                    default=0,
                    help="Number of frames in chunk")
parser.add_argument("--model_type",
                    type=str,
                    default="TabAttention",
                    help='Model type, one of: "RESNET3D", "DAFT", "INTERACTIVE", "TabAttention"')
parser.add_argument("--msha3D",
                    type=bool,
                    default=True,
                    help='Add MSHA to ResNet3D to create BabyNet')
parser.add_argument("--tabular_branch",
                    type=bool,
                    default=False,
                    help="Add tabular branch")
parser.add_argument("--normalize_tabular",
                    type=bool,
                    default=True,
                    help="Normalize tabular data.")
parser.add_argument("--tab_attention",
                    type=bool,
                    default=True,
                    help="Add tab attetion to T-CBAM")
parser.add_argument("--temporal_attention",
                    type=bool,
                    default=True,
                    help="Use temporal attention")
parser.add_argument("--temporal_mhsa",
                    type=bool,
                    default=True,
                    help="Temporal Attention computed with MHSA")
parser.add_argument("--cam_sam",
                    type=bool,
                    default=True,
                    help="Add CAM and SAM to CBAM")

args = parser.parse_args()

torch.manual_seed(0)


class DummyExperiment:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_parameters(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass

    def log_current_epoch(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass

    def train(self):
        return self


experiment = DummyExperiment()
experiment.log_parameters(args)

dataset = FetalWeightVideo(input_path=args.data,
                           x_image_size=args.x_img_size,
                           y_image_size=args.y_img_size,
                           pixels_crop=args.pixels_crop,
                           skip_frames=args.skip_frames,
                           n_frames=args.frames_num,
                           normalize_tabular=args.normalize_tabular)

if args.GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def calculate_metrics(y_true, y_pred, log=False, suffix="train", return_metrics=False):
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("")
    print("___________________")
    print(f"RESULTS: {suffix}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.4f}")
    print("___________________")
    if log:
        metrics = {
            f"rmse_{suffix}\n": rmse,
            f"mse_{suffix}\n": mse,
            f"mae_{suffix}\n": mae,
            f"mape_{suffix}\n": mape,
        }
        experiment.log_metrics(metrics)
    if return_metrics:
        return mse, rmse, mae, mape


def average_list(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def analyze_losses_per_patient(prediction_table):
    df = prediction_table.groupby(["patient_id"]).mean().reset_index()
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    mse, rmse, mae, mape = calculate_metrics(y_true, y_pred, log=True, suffix="val_per_patient", return_metrics=True)
    return mae

sgkfold = StratifiedGroupKFold(n_splits=5, shuffle=False)

criterion_reg = nn.MSELoss()

train_dataset = FetalWeightVideo(input_path=args.data,
                                 x_image_size=args.x_img_size,
                                 y_image_size=args.y_img_size,
                                 pixels_crop=args.pixels_crop,
                                 skip_frames=args.skip_frames,
                                 n_frames=args.frames_num,
                                 normalize_tabular=args.normalize_tabular,
                                 mode="train")
val_dataset = FetalWeightVideo(input_path=args.data,
                               x_image_size=args.x_img_size,
                               y_image_size=args.y_img_size,
                               pixels_crop=args.pixels_crop,
                               skip_frames=args.skip_frames,
                               n_frames=args.frames_num,
                               normalize_tabular=args.normalize_tabular,
                               mode="val")

print("---------------")

loss_min = np.inf

# Start time of learning
total_start_training = time.time()

model_types = ["RESNET3D", "DAFT", "INTERACTIVE", "TabAttention"]
assert args.model_type in model_types

best_results = {}

for fold, (train_ids, valid_ids) in enumerate(
        sgkfold.split(dataset, dataset.class_weight_by_chunk, dataset.patient_id_by_chunk)):
    print(f"FOLD {fold}")
    print("----------------")

    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_subsampler,
                              num_workers=args.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              sampler=valid_subsampler,
                              num_workers=args.num_workers,
                              pin_memory=True)

    # Init neural network
    if args.model_type == "RESNET3D":
        model = r3d_18(msha=args.msha3D, n_frames=args.frames_num,
                       input_size=(args.y_img_size - args.pixels_crop, args.x_img_size - args.pixels_crop))
    elif args.model_type == "DAFT":
        model = DAFT(in_channels=1, n_outputs=1)
    elif args.model_type == "INTERACTIVE":
        model = InteractiveHNN(in_channels=1, n_outputs=1, ndim_non_img=6)
    elif args.model_type == "TabAttention":
        model = ResNetTabAttention(cbam=True, tabattention=args.tab_attention,
                                   input_size=(args.y_img_size - args.pixels_crop, args.x_img_size - args.pixels_crop),
                                   n_frames=args.frames_num, temporal_mhsa=args.temporal_mhsa,
                                   temporal_attention=args.temporal_attention, tabular_branch=args.tabular_branch,
                                   cam_sam=args.cam_sam)
    else:
        raise NotImplementedError(f"No model for model type: {args.model_type}")

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, verbose=True)

    best_val_score = np.inf
    best_val_preds = None
    with experiment.train():
        for epoch in range(args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            model.train()
            running_loss = 0.0

            y_true = []
            y_pred = []
            patient_running_loss = []
            for batch_idx, (videos, weights, numerical_features, patient_id, body_part, first_frame) in enumerate(
                    train_loader):
                optimizer.zero_grad()
                videos = torch.permute(videos, (0, 1, 4, 2, 3))
                if args.model_type in ["RESNET3D", "DAFT", "INTERACTIVE", "TabAttention"]:
                    videos = torch.permute(videos, (0, 2, 1, 3, 4))

                videos = videos.to(device=device).float()
                y_true.extend(weights.flatten().tolist())
                weights = weights.to(device=device).float()
                numerical_features = numerical_features.to(device=device).float()

                reg_out = model(videos, numerical_features)
                y_pred.extend(reg_out.flatten().cpu().tolist())
                loss_reg = criterion_reg(reg_out, weights)
                loss = loss_reg
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % args.display_steps == 0:
                    print('    ', end='')
                    print(f"Batch: {batch_idx + 1}/{len(train_loader)} "
                          f"Loss: {loss.item():.4f} "
                          f"Learning time: {(time.time() - start_time_epoch):.2f}s "
                          f"First frame: {first_frame[0]}")

            # evalute
            calculate_metrics(y_true, y_pred, log=True)
            print(f"Finished epoch {epoch + 1}, starting evaluation.")
            model.eval()
            val_running_loss = 0.0
            y_true = []
            y_pred = []
            folds = []
            body_parts = []
            patient_ids = []
            curr_patient_id = None
            for batch_idx, (videos, weights, numerical_features, patient_id, body_part, first_frame) in enumerate(
                    valid_loader):
                if curr_patient_id is None:
                    curr_patient_id = patient_id

                if curr_patient_id != patient_id:
                    curr_patient_id = patient_id

                videos = torch.permute(videos, (0, 1, 4, 2, 3))
                if args.model_type in ["RESNET3D", "DAFT", "INTERACTIVE", "TabAttention"]:
                    videos = torch.permute(videos, (0, 2, 1, 3, 4))
                videos = videos.to(device=device).float()
                y_true.extend(weights.flatten().tolist())
                body_parts.extend(list(body_part))
                folds.extend([fold] * len(body_part))
                patient_ids.extend(list(patient_id))
                weights = weights.to(device=device).float()
                numerical_features = numerical_features.to(device=device).float()

                reg_out = model(videos, numerical_features)
                y_pred.extend(reg_out.flatten().cpu().tolist())
                loss_reg = criterion_reg(reg_out, weights)
                loss = loss_reg

                val_running_loss += loss.item()

            calculate_metrics(y_true, y_pred, log=True, suffix="val")

            train_loss = running_loss / len(train_loader)
            val_loss = val_running_loss / len(valid_loader)

            table = [["patient_id", "y_true", "y_pred", "body_part", "fold"]]
            table.extend(list(zip(patient_ids, y_true, y_pred, body_parts, folds)))
            df_table = pd.DataFrame(
                {"patient_id": patient_ids, "y_true": y_true, "y_pred": y_pred, "body_part": body_parts, "fold": folds})
            mae_per_patient = analyze_losses_per_patient(df_table)
            if best_val_score > mae_per_patient:
                save_path = f"{args.model_name}-{args.model_type}-fold-{fold}_{args.x_img_size}.pt"
                torch.save(model.state_dict(), save_path)
                best_val_score = mae_per_patient
                best_val_preds = y_pred
                print(f"Current best val score {best_val_score}. Model saved!")
                experiment.log_table(f"best_val_preds_fold_{fold}.csv", table)
                best_results[fold] = df_table

            scheduler.step()

            experiment.log_current_epoch(epoch)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("val_loss", val_loss)

            print('    ', end='')
            print(f"Train Loss: {train_loss:.3f} "
                  f"Val Loss: {val_loss:.3f}")

        print(f"Fold {fold} finished!")
        save_fold_metrics(experiment, best_results[fold], fold)

with experiment.train():
    save_metrics_all(experiment, best_results)
    save_per_patient_results(experiment, best_results)

print('Training finished, took {:.2f}s'.format(time.time() - total_start_training))