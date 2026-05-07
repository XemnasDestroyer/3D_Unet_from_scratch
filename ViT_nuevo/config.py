NUM_EPOCHS = 10

IN_CHANNELS = 4
OUT_CHANNELS = 4
BACKGROUND_AS_CLASS = False

TRAIN_CUDA = True
# Peso de la clase minoritária/positva (tumor) en la función de pérdida BCE. 
#   Se puede ajustar según el desequilibrio de clases.
BCE_WEIGHT = 250

ROI_SIZE = (64, 64, 64)


# Transformaciones para CARGAR ambos (imagen y máscara real).
# - Compose: Permite encadenar varias transformaciones en un solo paso.
# - LoadImaged: Carga los archivos .nii y los convierte en tensores.
# - EnsureChannelFirstd: Asegura que los datos tengan la forma [Canal, D, H, W].
# - ScaleIntensityd: Normaliza la intensidad de las imágenes (no se aplica 
#       a las máscaras).
# - RandCropByPosNegLabeld: Recorta aleatoriamente parches de la imagen, 
#       dando prioridad a las regiones con etiquetas (pos) sobre las 
#       sin etiquetas (neg).
# - ToTensord: Convierte los datos a tensores de PyTorch.
# - Lambdad: Permite aplicar una función personalizada a los datos.
# - RandRotated, RandFlipd, RandGaussianNoised: Transformaciones de 
#       aumento de datos (data augmentation) para hacer el modelo más variado.
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    RandCropByPosNegLabeld, ToTensord, Lambdad,
    Orientationd, NormalizeIntensityd, SpatialPadd,
    RandRotated, RandFlipd, RandGaussianNoised)

# ------- Definimos las transformaciones -------
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    # BraTS ya suele venir con canales, si no, usa EnsureChannelFirstd
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    # Normalización específica para MRI (No usar ScaleIntensityRanged de CT)
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
    # Crop que asegure que caiga tumor en los parches
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1, neg=1,
        num_samples=8,
    ),
    ToTensord(keys=["image", "label"]),
])

# Transforms para Validación
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

# Usamos las mismas claves para que la máscara real esté alineada con la imagen
predict_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"])
])
# ------- Fin de las transformaciones -------

# ------- Configuración del dispositivo, modelo, optimizador y función de pérdida -------
# Importamos el modelo UNet3D que definimos en unet.py. Este modelo es una
# arquitectura de red neuronal convolucional diseñada 
# para segmentación de imágenes 3D.
from SwinUNetR import SwinUNETR
# La función de pérdida con pesos para manejar el desbalance de clases. Aplica una importancia
#   distinta a las distintas clases (fondo vs tumor) para que la red no se "olvide" de aprender a segmentar el tumor,
#   que es la clase minoritaria.# Librería principal de Deep Learning. Proporciona los tensores, 
# operaciones matemáticas y la funcionalidad de GPU.
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SwinUNETR(
    img_size=ROI_SIZE,
    in_channels=IN_CHANNELS,            # 1 porque solo cargamos FLAIR de momento
    out_channels=OUT_CHANNELS,           # Típico en BraTS (TC, WT, ET)
    feature_size=48,          # Tamaño base de las características
    use_checkpoint=True
).to(device)

# Optimizador (ajusta los pesos de la red)
# Deberia porbar con AdamW, pero Adam es un buen punto de partida
# Contiene los algoritmos que "aprenden", como Adam. Es el que ajusta los
# pesos de la red durante el entrenamiento.
import torch.optim as optim
optimizer = optim.Adam(model.parameters())

# alpha=0.2 (peso a los falsos negativos)
# beta=0.8 (peso a los falsos positivos -> ¡Esto es lo que evita que sea vaga!)
# Para calcular la métrica de TverskyLoss, que es una medida de solapamiento 
#   entre la máscara real y la predicha. Nos dico qué tan lejos está
#   nuestra predicción de la realidad.
from monai.losses import TverskyLoss
loss_function = TverskyLoss(sigmoid=True, alpha=0.5, beta=0.5)

# Contiene los bloques de construcción de las redes neuronales (capas, 
# activaciones, etc.)
from monai.losses import DiceLoss
pos_weight = torch.tensor([BCE_WEIGHT], dtype=torch.float32).to(device)
criterion = DiceLoss(to_onehot_y=4, softmax=True)

# Scheduler: Reduce el LR cuando el loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.5, verbose=True)    

# Para convertir la salida de la red (que es un valor continuo entre 0 y 1)
#   en una máscara binaria (0 o 1) usando un umbral (NO LO USO).
# from monai.transforms import AsDiscrete
# post_pred = AsDiscrete(threshold=0.5) # Umbral de 0.5 para ser más estrictos
# ------- Fin de la configuración -------