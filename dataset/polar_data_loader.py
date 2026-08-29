"""Dataset support for 0/45/90-degree polarization triplets.

Each polarization image is converted to grayscale and used as one input
channel, in the fixed order [0 degrees, 45 degrees, 90 degrees].  The target
remains an RGB image.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
POLARIZATION_ANGLES = ("0", "45", "90")


def _images_by_stem(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    images: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(
                    f"Duplicate scene id {path.stem!r} in {directory}: "
                    f"{images[path.stem].name} and {path.name}"
                )
            images[path.stem] = path
    return images


def discover_scene_ids(data_root: str | Path) -> List[str]:
    """Return naturally sortable scene ids that exist in all four folders."""
    root = Path(data_root)
    folders = [root / "input" / angle for angle in POLARIZATION_ANGLES]
    folders.append(root / "gt")
    mappings = [_images_by_stem(folder) for folder in folders]

    expected = set(mappings[0])
    mismatches = []
    for folder, mapping in zip(folders[1:], mappings[1:]):
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        if missing or extra:
            mismatches.append(
                f"{folder}: missing={missing[:10]}, extra={extra[:10]}"
            )
    if mismatches:
        raise ValueError("Polarization/GT scene ids do not match:\n" + "\n".join(mismatches))

    def sort_key(scene_id: str) -> Tuple[int, int | str]:
        return (0, int(scene_id)) if scene_id.isdigit() else (1, scene_id)

    return sorted(expected, key=sort_key)


def split_scene_ids(
    data_root: str | Path,
    num_scenes: int,
    val_scenes: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Select ``num_scenes`` deterministically, then split train/validation."""
    scene_ids = discover_scene_ids(data_root)
    if num_scenes > len(scene_ids):
        raise ValueError(
            f"Requested {num_scenes} scenes, but only {len(scene_ids)} complete scenes exist"
        )
    if not 0 < val_scenes < num_scenes:
        raise ValueError("VAL_SCENES must be greater than 0 and smaller than NUM_SCENES")

    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    selected = scene_ids[:num_scenes]
    return selected[:-val_scenes], selected[-val_scenes:]


class PolarizationDataset(Dataset):
    """Load aligned polarization triplets and RGB ground truth images."""

    def __init__(
        self,
        data_root: str | Path,
        scene_ids: Sequence[str],
        patch_size: Sequence[int],
        training: bool,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.scene_ids = list(scene_ids)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.training = training

        angle_maps = {
            angle: _images_by_stem(self.root / "input" / angle)
            for angle in POLARIZATION_ANGLES
        }
        gt_map = _images_by_stem(self.root / "gt")
        self.records = []
        for scene_id in self.scene_ids:
            try:
                inputs = tuple(angle_maps[angle][scene_id] for angle in POLARIZATION_ANGLES)
                target = gt_map[scene_id]
            except KeyError as exc:
                raise ValueError(f"Incomplete scene {scene_id!r}") from exc
            self.records.append((scene_id, inputs, target))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _augment(tensors: Iterable[torch.Tensor], transform_id: int):
        tensors = list(tensors)
        if transform_id == 1:
            return [tensor.flip(1) for tensor in tensors]
        if transform_id == 2:
            return [tensor.flip(2) for tensor in tensors]
        if transform_id == 3:
            return [torch.rot90(tensor, 1, (1, 2)) for tensor in tensors]
        if transform_id == 4:
            return [torch.rot90(tensor, 2, (1, 2)) for tensor in tensors]
        if transform_id == 5:
            return [torch.rot90(tensor, 3, (1, 2)) for tensor in tensors]
        if transform_id == 6:
            return [torch.rot90(tensor.flip(1), 1, (1, 2)) for tensor in tensors]
        if transform_id == 7:
            return [torch.rot90(tensor.flip(2), 1, (1, 2)) for tensor in tensors]
        return tensors

    def __getitem__(self, index: int):
        _, input_paths, target_path = self.records[index]

        # One grayscale image per polarization angle -> exactly three channels.
        polar_channels = []
        for path in input_paths:
            with Image.open(path) as image:
                polar_channels.append(TF.to_tensor(image.convert("L")))
        input_tensor = torch.cat(polar_channels, dim=0)
        with Image.open(target_path) as image:
            target_tensor = TF.to_tensor(image.convert("RGB"))

        if input_tensor.shape[1:] != target_tensor.shape[1:]:
            raise ValueError(
                f"Input/target size mismatch for scene {self.scene_ids[index]}: "
                f"{tuple(input_tensor.shape)} vs {tuple(target_tensor.shape)}"
            )

        patch_h, patch_w = self.patch_size
        image_h, image_w = target_tensor.shape[1:]
        if image_h < patch_h or image_w < patch_w:
            input_tensor = TF.resize(input_tensor, self.patch_size, antialias=True)
            target_tensor = TF.resize(target_tensor, self.patch_size, antialias=True)
            image_h, image_w = self.patch_size

        if self.training:
            top = random.randint(0, image_h - patch_h)
            left = random.randint(0, image_w - patch_w)
        else:
            top = (image_h - patch_h) // 2
            left = (image_w - patch_w) // 2

        input_tensor = input_tensor[:, top : top + patch_h, left : left + patch_w]
        target_tensor = target_tensor[:, top : top + patch_h, left : left + patch_w]

        if self.training:
            input_tensor, target_tensor = self._augment(
                (input_tensor, target_tensor), random.randint(0, 7)
            )

        return input_tensor, target_tensor
