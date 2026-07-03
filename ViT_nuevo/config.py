# =========================
# CONFIGURACIÓN GENERAL
# =========================

NUM_EPOCHS = 100  # Para entrenamiento final. Para prueba rápida puedes usar 1 o 2.

IN_CHANNELS = 4          # Modalidades MRI del dataset
OUT_CHANNELS = 4         # Fondo + 3 clases tumorales
INCLUDE_BACKGROUND = False

TRAIN_CUDA = True
ROI_SIZE = (64, 64, 64)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1

# Número de parches extraídos por muestra.
# Si el entrenamiento va muy lento, prueba con 4 en vez de 8.
NUM_SAMPLES = 4

VAL_INTERVAL = 1

# =========================
# IMPORTS
# =========================

import torch
import torch.optim as optim

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ToTensord,
)

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from SwinUNetR import SwinUNETR


# =========================
# DISPOSITIVO
# =========================

device = torch.device("cuda" if torch.cuda.is_available() and TRAIN_CUDA else "cpu")
print(f"Device utilizado: {device}")

if torch.cuda.is_available() and TRAIN_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================
# TRANSFORMACIONES
# =========================

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),

    EnsureChannelFirstd(keys=["image", "label"]),

    Orientationd(keys=["image", "label"], axcodes="RAS"),

    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    SpatialPadd(
        keys=["image", "label"],
        spatial_size=ROI_SIZE
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1,
        neg=1,
        num_samples=NUM_SAMPLES,
    ),

    # Aumentos de datos sencillos. Puedes quitarlos si quieres una prueba rápida.
    RandFlipd(
        keys=["image", "label"],
        spatial_axis=0,
        prob=0.5
    ),

    RandFlipd(
        keys=["image", "label"],
        spatial_axis=1,
        prob=0.5
    ),

    RandFlipd(
        keys=["image", "label"],
        spatial_axis=2,
        prob=0.5
    ),

    RandGaussianNoised(
        keys="image",
        prob=0.15,
        mean=0.0,
        std=0.01
    ),

    ToTensord(keys=["image", "label"]),
])


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),

    EnsureChannelFirstd(keys=["image", "label"]),

    Orientationd(keys=["image", "label"], axcodes="RAS"),

    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True
    ),

    ToTensord(keys=["image", "label"]),
])


# Para predicción conviene usar el mismo preprocesamiento que en validación.
predict_transform = val_transforms


# =========================
# MODELO
# =========================

model = SwinUNETR(
    img_size=ROI_SIZE,
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    feature_size=48,
    use_checkpoint=True
).to(device)


# =========================
# OPTIMIZADOR
# =========================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================
# FUNCIÓN DE PÉRDIDA
# =========================

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=INCLUDE_BACKGROUND
)


# =========================
# MÉTRICAS Y POSTPROCESADO
# =========================

dice_metric = DiceMetric(
    include_background=INCLUDE_BACKGROUND,
    reduction="mean_batch",
    get_not_nans=True
)

post_pred = AsDiscrete(
    argmax=True,
    to_onehot=OUT_CHANNELS
)

post_label = AsDiscrete(
    to_onehot=OUT_CHANNELS
)


# =========================
# SCHEDULER
# =========================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",        # max porque monitorizaremos Dice de validación
    patience=5,
    factor=0.5
)