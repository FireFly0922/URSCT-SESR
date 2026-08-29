# Polarization training on a Linux GPU server

The prepared experiment uses 20 complete scenes: 16 for training and 4 for
validation.  Every input is `[0-degree grayscale, 45-degree grayscale,
90-degree grayscale]`; the target is RGB.  The split is deterministic and is
recorded in `exps/polar_20scenes_Enh/split_manifest.json`.

## 1. Check the server and enter the repository

```bash
nvidia-smi
cd /path/to/URSCT-SESR
```

The repository must contain this layout:

```text
dataset/Polar_data/
├── input/
│   ├── 0/      # 151 images
│   ├── 45/     # 151 images
│   └── 90/     # 151 images
└── gt/         # 151 images
```

The extensions may be mixed (`.bmp` and `.png` are both supported), but the
four files in one scene must have the same filename stem, such as `015`.

## 2. Create a project environment

Python 3.10 and the pinned PyTorch 2.4.1 stack are a conservative combination
for this repository.  With an NVIDIA driver that supports CUDA 12.x, use:

```bash
conda create -n ursct-polar python=3.10 -y
conda activate ursct-polar
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server.txt
```

If the driver is too old for CUDA 12.1 but supports CUDA 11.8, replace `cu121`
with `cu118`.  PyTorch wheels include the CUDA runtime needed by PyTorch; a
separate system CUDA toolkit is not needed for this training script.

Verify that this is a CUDA build rather than a CPU-only build:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The third value must be `True`.

## 3. Run one complete smoke-test step

This performs one forward/backward training step and one validation forward
step.  It catches data, dependency, shape, CUDA, and memory problems before the
long run.

```bash
python -u scripts/Polar_train.py \
  --config configs/Polar_Enh_opt.yaml \
  --smoke-test
```

Expected key messages include:

```text
input_channels=[0_gray, 45_gray, 90_gray]; target=RGB
train_scenes=16
val_scenes=4
Smoke test passed
```

If CUDA reports out-of-memory, first change `OPTIM.BATCH` from `2` to `1` in
`configs/Polar_Enh_opt.yaml`.  If necessary, also change
`MODEL_DETAIL.USE_CHECKPOINTS` to `true`.

## 4. Start the 200-epoch run

`tmux` is recommended because the job survives an SSH disconnect:

```bash
tmux new -s ursct-polar
conda activate ursct-polar
cd /path/to/URSCT-SESR
python -u scripts/Polar_train.py \
  --config configs/Polar_Enh_opt.yaml \
  2>&1 | tee exps/polar_20scenes_Enh/train.log
```

Detach with `Ctrl-b`, then `d`.  Reattach later with:

```bash
tmux attach -t ursct-polar
```

The run writes:

```text
exps/polar_20scenes_Enh/
├── split_manifest.json
├── train.log
├── log/                 # TensorBoard events
├── results/             # input | prediction | GT previews
└── models/
    ├── model_latest.pth
    ├── model_bestPSNR.pth
    └── model_bestSSIM.pth
```

Validation runs every 5 epochs.  `model_latest.pth` is replaced every epoch,
so the disk is not filled with 200 full checkpoints.

## 5. Resume after interruption

```bash
conda activate ursct-polar
cd /path/to/URSCT-SESR
python -u scripts/Polar_train.py \
  --config configs/Polar_Enh_opt.yaml \
  --resume \
  2>&1 | tee -a exps/polar_20scenes_Enh/train.log
```

With no path after `--resume`, the script loads
`exps/polar_20scenes_Enh/models/model_latest.pth`, including optimizer,
scheduler, AMP scaler, best metrics, and the next epoch number.

## 6. Watch progress

Terminal monitoring:

```bash
tail -f exps/polar_20scenes_Enh/train.log
watch -n 1 nvidia-smi
```

TensorBoard through an SSH tunnel:

```bash
# On the server
tensorboard --logdir exps/polar_20scenes_Enh/log --host 127.0.0.1 --port 6006

# On the local computer
ssh -L 6006:127.0.0.1:6006 USER@SERVER
```

Then open `http://127.0.0.1:6006` locally.
