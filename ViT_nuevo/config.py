# =========================
# CONFIGURACIÓN GENERAL
# =========================

NUM_EPOCHS = 100

IN_CHANNELS = 4 
OUT_CHANNELS = 4
INCLUDE_BACKGROUND = False

TRAIN_CUDA = True
ROI_SIZE = (64, 64, 64)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1

NUM_SAMPLES = 4
VAL_INTERVAL = 5 

# =========================
# SELECCIÓN DE MODELO
# =========================
# Opciones:
#   "swinunetr" -> modelo principal del TFG
#   "unet3d"    -> baseline 3D U-Net

MODEL_NAME = "swinunetr"
# MODEL_NAME = "unet3d"

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

from monai.networks.nets import UNet, SwinUNETR


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
# CREACIÓN DEL MODELO
# =========================

def create_model(model_name: str):
    """
    Crea el modelo seleccionado.
    Permite alternar entre Swin UNETR y 3D U-Net sin pisar configuraciones.
    """

    model_name = model_name.lower()

    if model_name == "swinunetr":
        print("[INFO] Modelo seleccionado: Swin UNETR")

        return SwinUNETR(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=48,
            use_checkpoint=True
        )

    elif model_name == "unet3d":
        print("[INFO] Modelo seleccionado: 3D U-Net baseline")

        return UNet(
            spatial_dims=3,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    else:
        raise ValueError(
            f"Modelo no reconocido: {model_name}. "
            "Usa 'swinunetr' o 'unet3d'."
        )

model = create_model(MODEL_NAME).to(device)

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
    mode="max",
    patience=5,
    factor=0.5
)