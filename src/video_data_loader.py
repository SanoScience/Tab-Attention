import glob
import os
import re

import cv2
import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import groupby
from collections import defaultdict


class FetalWeightVideo(Dataset):
    """
    FetalWeightVideo class.
    """

    def __init__(self,
                 input_path: str = None,
                 resample: bool = True,
                 n_frames: int = 48,
                 x_image_size: int = 512,
                 y_image_size: int = 384,
                 pixels_crop: int = 0,
                 max_width: int = 1680,
                 max_height: int = 1260,
                 skip_frames: int = 0,
                 multiple_planes: bool = False,
                 background_frames_filename: str = None,
                 resize_only: bool = False,
                 padding: bool = False,
                 normalize_tabular: bool = False,
                 mode: str = None) -> None:
        """

        Args:
            input_path:
            resample:
            n_frames: if multiple_planes, n_frames must be divisible by 3
            x_image_size: the assumption is that x>y and x/y= ~0.75
            y_image_size:
            pixels_crop:
            max_width:
            max_height:
            skip_frames:
            multiple_planes:
            background_frames_filename:
            resize_only:
            padding:
            mode:
        """
        self.input_path = os.path.join(input_path, "frames")
        self.n_frames = n_frames
        self.x_image_size = x_image_size
        self.y_image_size = y_image_size
        self.max_width = max_width
        self.max_height = max_height
        self.pixels_crop = pixels_crop
        self.multiple_planes = multiple_planes
        self.labels = pd.read_csv("../data/clinical.csv", sep=";", usecols=["ID", "AC", "HC", "BPD", "FL",
                                                                                      "Age", "GA", "Weight",
                                                                                      "Weight_class"])
        self.resize_only = resize_only

        if normalize_tabular:
            self.normalize_labels()

        assert (self.resize_only and pixels_crop == 0) or (not self.resize_only)
        assert mode in ["train", "val", None]
        self.mode = mode
        self.patient_id = []
        for file in os.listdir(self.input_path):
            if not file.startswith("."):
                pid = file.split("_")[1]
                self.patient_id.append(pid)

        self.resample = resample

        self.chunk_frames = []
        self.patient_id_by_chunk = []
        self.class_weight_by_chunk = []
        self.frames_to_omit = []
        if background_frames_filename:
            background_frames = pd.read_csv(os.path.join(input_path, background_frames_filename))
            background_frames = background_frames["background_frames"].tolist()
            background_frames = [f"{f}.png" for f in background_frames]
            self.frames_to_omit = background_frames

        for i in self.patient_id:
            frames = glob.glob(f"{self.input_path}/video_{i}/*abdomen*.png")
            frames = sorted(frames, key=numericalSort)
            frames = [f for f in frames if self._get_filename(f) not in self.frames_to_omit]
            if multiple_planes:
                chunk_frames = self.chunk_frames_multiple_planes(frames, n_frames)
            else:
                chunk_frames = chunks(frames, n_frames, skip_frames)

            for chunk in chunk_frames:
                if len(chunk) != n_frames and not padding:
                    continue
                chunk.extend(["padding" for _ in range(n_frames - len(chunk))])
                class_weight_by_chunk = self.labels.loc[self.labels["ID"] == int(i), "Weight_class"].item()
                self.chunk_frames.append([chunk, i])
                self.patient_id_by_chunk.append(i)
                self.class_weight_by_chunk.append(class_weight_by_chunk)

    def normalize_labels(self):
        for column in ["AC", "HC", "BPD", "FL", "Age", "GA"]:
            mean = self.labels[column].mean()
            std = self.labels[column].std()
            print(f"Normalizing {column} with mean: {mean} and std: {std}")
            self.labels[column] = (self.labels[column] - mean) / std

    def __getitem__(self, x):
        """

        Args:
            x:

        Returns:

        """
        chunk_frames, patient_id = self.chunk_frames[x][0], self.chunk_frames[x][1]
        patient_dir = f"{self.input_path}/video_{patient_id}"
        video_stack = self.load_video(patient_dir, chunk_frames, resample=True)
        weight = self.labels.loc[self.labels["ID"] == int(patient_id), "Weight"]
        weight = weight.item()

        numerical_features = self.labels.loc[self.labels["ID"] == int(patient_id), ["AC", "HC", "BPD", "FL", "Age",
                                                                                    "GA"]]
        numerical_features = np.array(numerical_features)
        numerical_features = torch.tensor(numerical_features, dtype=torch.float)

        video_stack = torch.tensor(video_stack)

        body_part = self.get_body_part(chunk_frames)
        first_frame = self._first_frame(chunk_frames)
        return video_stack, weight, numerical_features, self.patient_id_by_chunk[x], body_part, first_frame

    def __len__(self) -> int:
        return len(self.chunk_frames)

    def _get_filename(self, full_path):
        return os.path.basename(full_path)

    def get_body_part(self, chunk_frames):
        body_parts = [self._body_part_from_frame_path(frame) for frame in chunk_frames if "padding" not in frame]
        return max(body_parts, key=body_parts.count)

    def _body_part_from_frame_path(self, path):
        body_part = path.split('video_')[-1].split("_")[-2]
        if "abdomen" in body_part:
            return "abdomen"
        if "head" in body_part:
            return "head"
        if "femur" in body_part:
            return "femur"

        raise ValueError(f"Unrecognized body part: {body_part}")

    def load_video(self, patient_dir: str = None,
                   chunk_frames: list = None,
                   resample: bool = True):
        """

        Args:
            resample:
            chunk_frames:
            patient_dir:

        Returns:

        """
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(patient_dir)

        video = np.zeros(
            (len(chunk_frames), self.y_image_size - self.pixels_crop, self.x_image_size - self.pixels_crop, 1),
            np.float32)
        video_frames = {}
        for count, chunk in enumerate(chunk_frames):
            if resample:
                if "padding" in chunk:
                    continue
                frame = cv2.imread(chunk, 0)

                if self.resize_only:
                    transformations = A.Compose([
                        A.Resize(self.y_image_size, self.x_image_size),
                    ])
                else:
                    height, width = frame.shape[:2]
                    transformed_height = int((height / self.max_width) * self.x_image_size)
                    transformed_width = int((width / self.max_width) * self.x_image_size)

                    assert transformed_width <= self.x_image_size
                    assert transformed_height <= self.y_image_size
                    transformations = A.Compose([
                        A.Resize(transformed_height, transformed_width),
                        A.PadIfNeeded(self.y_image_size, self.x_image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                        A.CenterCrop(self.y_image_size - self.pixels_crop, self.x_image_size - self.pixels_crop),
                    ])

                transformed_frame = transformations(image=frame)
                transformed_frame = transformed_frame["image"]
                transformed_frame = np.expand_dims(transformed_frame, axis=2)
                frame_name = f"image{count - 1}" if count != 0 else "image"
                video_frames[frame_name] = transformed_frame

        if self.mode == "val":
            augmentations = A.Compose([])
        else:
            augmentations = A.Compose([A.Rotate(limit=(-25, 25)),
                                       A.HorizontalFlip(p=0.5),
                                       A.RandomBrightnessContrast(),
                                       A.ImageCompression(p=0.1),
                                       A.OneOf([
                                           A.MotionBlur(p=0.5),
                                           A.MedianBlur(blur_limit=3, p=0.5),
                                           A.Blur(blur_limit=3, p=0.5),
                                           A.GaussianBlur(p=0.5)],
                                           p=0.5),
                                       ],
                                      additional_targets={f"image{i}": "image" for i in range(len(video_frames) - 1)})

        transformed_video = augmentations(**video_frames)
        for i, (frame_name, frame) in enumerate(video_frames.items()):
            augmented_frame = transformed_video[frame_name]
            video[i] = augmented_frame

        m = np.max(video)
        video = (video - 0.5 * m) / (0.5 * m)  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        return video

    def chunk_frames_multiple_planes(self, frames, n_frames):
        by_body_parts = groupby(frames, self._body_part_from_frame_path)
        body_parts_dict = defaultdict(list)
        max_length = 0
        for b_p, fr in by_body_parts:
            b_p_list = list(el for el in fr)
            if len(b_p_list) > max_length:
                max_length = len(b_p_list)
            body_parts_dict[b_p] = b_p_list
        max_length_with_padding = max_length + (n_frames // 3 - (max_length % (n_frames // 3)))
        abdomen_frames = self._pad(body_parts_dict["abdomen"], max_length_with_padding)
        femur_frames = self._pad(body_parts_dict["femur"], max_length_with_padding)
        head_frames = self._pad(body_parts_dict["head"], max_length_with_padding)
        chunk_frames = []
        body_part_frames_num = n_frames // 3
        for i in range(max_length_with_padding // (n_frames // 3)):
            new_chunk = []
            new_chunk.extend(abdomen_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            new_chunk.extend(femur_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            new_chunk.extend(head_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            all_padding = True
            for f in new_chunk:
                if "padding" not in f:
                    all_padding = False
            if all_padding:
                continue
            chunk_frames.append(new_chunk)

        return chunk_frames

    def _pad(self, l: list, pad_to: int):
        l.extend(["padding" for _ in range(pad_to - len(l))])
        return l

    def _first_frame(self, frames):
        for f in frames:
            if "padding" not in f:
                return "video" + f.split("video")[-1]


def chunks(L, n, skip):
    """

    Args:
        L:
        n:
        skip:

    Returns:

    """
    if skip > 0:
        return [L[x: x + (skip + 1) * (n - 1) + 1: skip + 1] for x in range(0, len(L), (skip + 1) * (n - 1) + 1)]
    else:
        return [L[x: x + n: skip + 1] for x in range(0, len(L), n)]


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fetal_video = FetalWeightVideo("../data/", x_image_size=512, y_image_size=512, n_frames=16, mode="train")
    print(len(fetal_video))
    for i, (video, weight, numerical_features, patient_id, body_part, first_frame) in enumerate(fetal_video):
        img = np.concatenate([v for v in video], axis=1)
        plt.imshow(img, cmap='gray')
        plt.show()
        print(i, video.shape, weight, numerical_features.shape, first_frame)
