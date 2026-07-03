import os
import csv
import random
from time import time

import torch
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.apps import DecathlonDataset
from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from torch.cuda.amp import autocast, GradScaler

from config import (
    NUM_EPOCHS,
    VAL_INTERVAL,
    BATCH_SIZE,
    ROI_SIZE,
    device,
    model,
    optimizer,
    scheduler,
    loss_function,
    dice_metric,
    post_pred,
    post_label,
    train_transforms,
    val_transforms
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_medical_volume():
    """
    Descarga y carga el dataset Task01_BrainTumour del Medical Segmentation Decathlon.
    Divide los sujetos etiquetados en entrenamiento y validación siguiendo una proporción 80/20.
    """

    root_dir = "./dataset_brats"
    os.makedirs(root_dir, exist_ok=True)

    # Descarga del dataset si no está disponible
    DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        download=True,
        cache_num=5,
    )

    list_dir = "./dataset_brats/Task01_BrainTumour"
    jsonlist = os.path.join(list_dir, "dataset.json")
    datadir = list_dir

    # Cargamos todos los sujetos etiquetados de training
    all_labeled_files = load_decathlon_datalist(
        jsonlist,
        is_segmentation=True,
        data_list_key="training",
        base_dir=datadir,
    )

    # División reproducible 80/20
    random.seed(42)
    random.shuffle(all_labeled_files)

    split_index = int(len(all_labeled_files) * 0.8)

    train_files = all_labeled_files[:split_index]
    val_files = all_labeled_files[split_index:]

    print("\n" + "=" * 50)
    print(f"[INFO] Total de sujetos etiquetados: {len(all_labeled_files)}")
    print(f"[INFO] Sujetos de entrenamiento: {len(train_files)}")
    print(f"[INFO] Sujetos de validación: {len(val_files)}")
    print("=" * 50 + "\n")

    # Puedes cambiar CacheDataset por Dataset si te da problemas de RAM.
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=0.2,
        num_workers=0,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=0.2,
        num_workers=0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def check_batch_shapes(train_loader):
    """
    Comprueba formas de imagen y etiqueta antes de entrenar.
    Es importante para verificar canales, clases y formato de las máscaras.
    """

    batch = next(iter(train_loader))

    print("\n" + "=" * 50)
    print("[CHECK] Forma de image:", batch["image"].shape)
    print("[CHECK] Forma de label:", batch["label"].shape)
    print("[CHECK] Valores únicos de label:", torch.unique(batch["label"]))
    print("=" * 50 + "\n")


def save_checkpoint(epoch, metric, filename="best_swinunetr_model.pth"):
    """
    Guarda el mejor modelo según Dice medio de validación.
    """

    checkpoint = {
        "epoch": epoch,
        "best_metric": metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)
    print(f"[INFO] Mejor modelo guardado en época {epoch} con Dice {metric:.4f}")


def validate(val_loader):
    """
    Evalúa el modelo sobre el conjunto de validación.
    Calcula Dice medio y Dice por clase.
    """

    model.eval()

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="Validación"):
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            with autocast(enabled=torch.cuda.is_available()):
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5
                )

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs, y=val_labels)

        dice_result, not_nans = dice_metric.aggregate()
        dice_metric.reset()

    # dice_result suele tener un valor por clase si reduction="mean_batch"
    dice_per_class = dice_result.cpu().numpy()

    # Si include_background=False, estas clases serán las tumorales.
    mean_dice = dice_per_class.mean()

    return mean_dice, dice_per_class


def train(train_loader, val_loader):
    """
    Entrenamiento principal del modelo.
    Guarda loss, Dice medio y Dice por clase en un CSV.
    """

    print("\nIniciando entrenamiento...\n")

    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_metric = -1.0
    best_metric_epoch = -1

    log_path = "training_log.csv"

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_mean_dice",
            "dice_class_1",
            "dice_class_2",
            "dice_class_3",
            "epoch_time_seconds",
        ])

    total_start = time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time()

        print("-" * 50)
        print(f"Época {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        epoch_loss = 0.0
        step_count = 0

        loop_batches = tqdm(train_loader, desc=f"Entrenamiento {epoch + 1}/{NUM_EPOCHS}")

        for batch_data in loop_batches:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1

            loop_batches.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / step_count if step_count > 0 else 0.0
        epoch_time = time() - epoch_start

        print(f"[TRAIN] Loss media: {train_loss:.4f}")
        print(f"[TIME] Tiempo época: {epoch_time / 60:.2f} minutos")

        # Validación cada VAL_INTERVAL épocas
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_mean_dice, dice_per_class = validate(val_loader)

            print(f"[VAL] Dice medio: {val_mean_dice:.4f}")
            print(f"[VAL] Dice por clase: {dice_per_class}")

            # Scheduler sobre Dice de validación
            scheduler.step(val_mean_dice)

            # Guardar logs
            dice_values = list(dice_per_class)

            # Aseguramos 3 clases tumorales aunque haya menos por algún motivo
            while len(dice_values) < 3:
                dice_values.append(float("nan"))

            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_mean_dice,
                    dice_values[0],
                    dice_values[1],
                    dice_values[2],
                    epoch_time,
                ])

            # Guardar mejor modelo
            if val_mean_dice > best_metric:
                best_metric = val_mean_dice
                best_metric_epoch = epoch + 1
                save_checkpoint(epoch + 1, best_metric)

    total_time = time() - total_start

    print("\nEntrenamiento finalizado.")
    print(f"Mejor Dice medio: {best_metric:.4f} en época {best_metric_epoch}")
    print(f"Tiempo total: {total_time / 3600:.2f} horas")
    print(f"Logs guardados en: {log_path}")


if __name__ == "__main__":
    train_loader, val_loader = load_medical_volume()

    check_batch_shapes(train_loader)

    train(train_loader, val_loader)