# Interactúa con el sistema operativo. Se usara principialmente para
# verificar si los archivos .nii existen antes de intentar abrirlos.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Para pasar argumentos desde la terminal (-help, train, predict).
import argparse 

# Torchsummary es una herramienta para mostrar un resumen de la arquitectura 
#   del modelo, incluyendo el número de parámetros y la forma de las salidas 
#   de cada capa.    
from torchsummary import summary
# CrossEntropyLoss nos permitirá cambiar el peso de cada clase para
#   manejar el desbalance de clases (muchos más píxeles de fondo 
#   que de tumor).
#from torch.nn import CrossEntropyLoss

# Biblioteca estándar para abrir, leer y escribir archivos de 
# imágenes médicas en formato NIfTI (.nii o .nii.gz)
import nibabel as nib

# - El Dataset organiza los diccioarios en imágenes
# - El DataLoader se encarga de crear los batches, mezclar los datos y 
#       cargarlos en GPU eficientemente
from monai.data import Dataset, DataLoader

# Para realizar la inferencia por ventanas deslizantes, que es una técnica
#   que permite segmentar imágenes grandes dividiéndolas en partes más pequeñas.
#   Luego se encarga de ensamblar las predicciones de cada parte para obtener
#   la segmentación completa.
from monai.inferers import sliding_window_inference

# Librería para visualizar los resultados en tiempo real durante 
#   el entrenamiento.
import matplotlib.pyplot as plt

# Configuración específica de la red neuronal
from config import (NUM_EPOCHS,
                    device, model, optimizer, criterion,
                    train_transforms, val_transforms)

from time import time

from tqdm import tqdm

import torch


def load_medical_volume():
    """
    Descarga el dataset BrainTS si no lo encuentra en el disco.

    Parameters
    ----------

    Returns
    -------
    dict
        A dictionary containing the paths to the image and label files.
    """  
    from monai.apps import DecathlonDataset


    root_dir = "./dataset_brats"
    os.makedirs(root_dir, exist_ok=True)
    ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        download=True,
        cache_num=5,     # Solo mantiene 5 imágenes en memoria RAM
    )

    # 1. Definimos las rutas de BraTS
    list_dir = "./dataset_brats/Task01_BrainTumour" # Donde está el dataset.json
    jsonlist = os.path.join(list_dir, "dataset.json")
    datadir = list_dir # Directorio base para las rutas del JSON
    
    num_workers = 0

    # 2. Cargamos las listas de archivos de BraTS
    # is_segmentation=True es clave para que cargue la ruta de la etiqueta

    from monai.data import load_decathlon_datalist
    train_files = load_decathlon_datalist(jsonlist, True, "training", base_dir=datadir)
    val_files = load_decathlon_datalist(jsonlist, True, "test", base_dir=datadir)
    
    print(f"BraTS Training: {len(train_files)} sujetos")
    print(f"BraTS Validation: {len(val_files)} sujetos") 
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, # En validación solemos usar 1 para evaluar el volumen completo
        num_workers=num_workers, 
        shuffle=False
    )

    return train_loader, val_loader

def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Carga un checkpoint guardado previamente para continuar el entrenamiento desde donde se dejó.
    Si no hay un checkpoint, devuelve 0 para empezar desde la época 1.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load the checkpoint into.
    optimizer : torch.optim.Optimizer
        The optimizer to load the checkpoint into.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The scheduler to load the checkpoint into.
    filename : str
        The path to the checkpoint file.

    Returns
    -------
    int
        The epoch from which to continue training.
    """

    if os.path.exists(filename):
        print(f"--> Cargando Checkpoint desde {filename}...")
        checkpoint = config.torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None: 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None: 
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    print(f"--> No se encontró el checkpoint en {filename}. Empezando desde cero.")
    return 0

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    """
    Guarda el estado actual del modelo, optimizador y scheduler en un checkpoint para poder continuar el entrenamiento más tarde.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        The optimizer to save.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The scheduler to save.
    epoch : int
        The current epoch number to save in the checkpoint.
    filename : str
        The path to the checkpoint file.
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    config.torch.save(checkpoint, filename)
    print(f"--> Checkpoint guardado en época {epoch}")

def train(train_loader):
    print("\nIniciando entrenamiento...")
    historial_errores = []

    MAX_STEPS_PER_EPOCH = 50

    for epoch in range(NUM_EPOCHS):
        model.train()
        error_epoca = 0.0
        pasos_reales = 0
        
        # tqdm ahora itera directamente sobre tu DataLoader
        loop_batches = tqdm(train_loader, desc=f"Época [{epoch+1}/{NUM_EPOCHS}]")
        
        for step, batch in enumerate(loop_batches):
            # Limitamos el número de pasos de entrenamiento para que no tarde 2 horas
            if step >= MAX_STEPS_PER_EPOCH:
                break
            # Extraemos las imágenes y las etiquetas de tu batch.
            # Ajusta las claves ('image', 'label') según cómo hayas definido tu DataLoader
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device) 
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            # Acumular el error
            error_epoca += loss.item()
            pasos_reales += 1
            
            # Actualizamos la barra de progreso
            loop_batches.set_postfix(error=f"{loss.item():.4f}")
            
        # Calcular el error medio de la época
        # len(train_loader) nos da el número total de batches
        error_medio = error_epoca / pasos_reales if pasos_reales > 0 else 0
        historial_errores.append(error_medio)
        
        print(f"Error medio de la Época {epoch+1}: {error_medio:.4f}\n")

    print("¡Entrenamiento finalizado!")

    # ==========================================
    # 4. Visualización con Matplotlib
    # ==========================================
    if historial_errores:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), historial_errores, marker='o', linestyle='-', color='b', label='Error de Entrenamiento')
        plt.title('Curva de Aprendizaje - Swin UNETR (BraTS)')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.xticks(range(1, NUM_EPOCHS + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

def predict(model_path, image_path, mask_path):
    model = config.model

    # Cargar los pesos y poner en modo evaluación
    load_checkpoint(model, None, None, model_path)
    model.eval() 

    # Preparar los datos
    data = predict_transform({"image": image_path, "label": mask_path})
    
    # Preparamos el tensor para la red (añadimos dimensión de batch)
    input_tensor = data["image"].unsqueeze(0).to(config.device) # [1, 1, D, H, W]
    
    with config.torch.no_grad():
        prediction = sliding_window_inference(
            inputs=input_tensor, 
            roi_size=(64, 64, 64), 
            sw_batch_size=4, 
            predictor=model,
            overlap=0.5
        )

    # Sigmoid + Umbral de 0.5 para binarizar
    prediction_binaria = (config.torch.sigmoid(prediction) > 0.5).float()

    # Convertir el tensor de predicción a un array de numpy
    # Quitamos las dimensiones extras de Batch y Canal para dejarlo en [D, H, W]
    pred_mask = prediction_binaria[0, 0].cpu().numpy() # [D, H, W]

    # Cargar la imagen original para copiar sus "metadatos"
    # Esto es vital para que 3D Slicer sepa dónde colocar la máscara
    original_nifti = nib.load(image_path)
    header = original_nifti.header
    affine = original_nifti.affine

    # Crear el nuevo objeto NIfTI
    # Nos aseguramos de que el tipo de dato sea compatible (int16 o uint8 suele bastar)
    pred_nifti = nib.Nifti1Image(pred_mask.astype("uint8"), affine, header)

    # Guardar en el disco
    output_path = "./assets/data/3d/brain/prediccion_3d_final.nii"
    nib.save(pred_nifti, output_path)

    print(f"¡Segmentación guardada con éxito en: {output_path}!")

def visualize_progress(lista_loss, lista_dice):
    # Crear una figura con dos columnas
    plt.figure(figsize=(14, 5))

    # Gráfica de la Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(lista_loss) + 1), lista_loss, label='Pérdida (Tversky)', color='tab:red', linewidth=2)
    plt.title('Progreso del Error (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Valor de Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Guardar la imagen en el disco
    plt.tight_layout()
    plt.savefig("entrenamiento_stats.png")
    print("\n[INFO] Gráficas guardadas como 'entrenamiento_stats.png'")
    plt.show()

if __name__ == "__main__":
    train_loader, test_loader = load_medical_volume()
    train(train_loader)