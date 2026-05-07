# Interactúa con el sistema operativo. Se usara principialmente para
# verificar si los archivos .nii existen antes de intentar abrirlos.
import os

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
from config import (BACKGROUND_AS_CLASS, IN_CHANNELS, NUM_CLASSES, BCE_WEIGHT, 
                    train_transform, predict_transform)
import config


def load_medical_volume(path):
    """
    Carga un volumen médico en formato NIfTI (.nii) y su máscara correspondiente, verificando que ambos archivos existan.

    Parameters
    ----------
    path : str
        The path to the medical volume files.

    Returns
    -------
    dict
        A dictionary containing the paths to the image and label files.
    """
    img_path = path + "brain_t1.nii"
    mask_path = path + "final_seg.nii"

    if not os.path.exists(img_path):
        raise ValueError("No se encuentra el archivo de imagen .nii. Revisa la ruta.")
    if not os.path.exists(mask_path):
        raise ValueError("No se encuentra el archivo de la máscara .nii. Revisa las rutas.")
    
    return {"image": img_path, "label": mask_path}

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

def train(device, data_path, model_path, NUM_EPOCHS):
    global NUM_CLASSES
    if BACKGROUND_AS_CLASS: NUM_CLASSES +=1

    # Llamada a la funcion que carga la imagen y la máscara, y devuelve un diccionario con las rutas de ambos.
    data_dict = load_medical_volume(data_path)   

    # DataDict va entre corchetes porque el Dataset de MONAI espera una lista de diccionarios, aunque aquí solo tenemos uno
    train_ds = Dataset(data=[data_dict], transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Cargar progreso si existe
    start_epoch = load_checkpoint(config.model, config.optimizer, config.scheduler, model_path)
    
    print(f"Iniciando entrenamiento en {device}...")

    # Listas para las gráficas de progreso
    history_loss = []
    history_dice = []
    
    print(f"Iniciando desde época {start_epoch+1} hasta {start_epoch + NUM_EPOCHS}")

    for epoch in range(NUM_EPOCHS):
        config.torch.cuda.empty_cache() # Libera memoria no utilizada en GPU al inicio de cada época
        config.model.train() # Activa el modo entrenamiento de la red
        epoch_loss = 0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            if epoch == 0:
                print(f"Valores únicos en la máscara: {config.torch.unique(labels)}")
                print(f"Forma del tensor de entrada: {inputs.shape}")

            config.optimizer.zero_grad() # Limpieza de gradientes viejos
            outputs = config.model(inputs) # Obtiene la predicción de la red para los inputs actuales
            
            # loss = loss_function(outputs, labels)
            loss_tversky = config.loss_function(outputs, labels)
            loss_bce = config.criterion(outputs, labels)
            loss = loss_tversky + loss_bce
            loss.backward() # Calcula los gradientes de la pérdida con respecto a los pesos de la red

            config.optimizer.step() # Actualiza los pesos de la red usando los gradientes calculados

            epoch_loss += loss.item()

            # --- VISUALIZACIÓN EN TIEMPO REAL ---
            # if (epoch + 1) % 20 == 0 and idx == 0:
            #     visualizar_progreso(inputs, labels, outputs, epoch + 1)

        # Calculamos el promedio del loss en esta época
        avg_loss = epoch_loss / len(train_loader)

        # Guardamos en el historial para las gráficas
        history_loss.append(avg_loss)

        # Activamos el scheduler para ajustar el LR si el loss se estanca
        config.scheduler.step(avg_loss)
        current_lr = config.optimizer.param_groups[0]['lr']

        print(f"EPOCH {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Guardar cada 50 épocas o al final
        # if (epoch + 1) % 50 == 0:
        #     save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)

    # Guardar la red neuronal
    save_checkpoint(config.model, config.optimizer, config.scheduler, epoch+1, model_path)
    print("Modelo guardado como unet3d_brain_model.pth")

    print("\n¡Entrenamiento finalizado!")

    visualize_progress(history_loss, history_dice)

    samples = config.train_transform(data_dict)
    print(f"Forma del tensor de imagen: {samples[3]['image'].shape}")

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

# --- Test de dimensionamiento ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de UNet 3D para Cerebro")

    parser.add_argument("mode", choices=["train","predict"], 
                        help="Selecciona 'train' para entrenar o " \
                        "'predict' para inferencia")

    # Opcional: --epochs (se puede abreviar como -e)
    # type=int asegura que el dato sea un número
    # default=100 es el valor si el usuario no pone nada
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Número de épocas para el entrenamiento (por defecto: 100)")

    args = parser.parse_args()
    from datetime import datetime
    fecha = datetime.now().strftime("%d%m%Y")
    model_path = f"./unet3d_brain_model_{fecha}.pth"
    
    if args.mode == "predict":
        predict(model_path=model_path, 
                image_path="./assets/data/3d/brain/00000057_brain_t1.nii",
                mask_path="./assets/data/3d/brain/00000057_final_seg.nii")
    
    elif args.mode == "train":
        # El numero de épocas es el número de veces que la red verá todo el conjunto de datos (en este caso, los parches generados)
        NUM_EPOCHS = args.epochs  # Empieza con pocas para probar
        data_path = "./assets/data/3d/brain/00000057_" # La ruta donde están los archivos .nii
        model_path = "./unet3d_brain_model_16032026.pth" # Ruta donde se guardará el modelo entrenado (checkpoint)
        train(config.device, data_path, model_path, NUM_EPOCHS)   