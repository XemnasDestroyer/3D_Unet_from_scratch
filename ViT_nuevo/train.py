import os
import csv
import random
from time import time

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.apps import DecathlonDataset
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from torch.amp import autocast, GradScaler

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
    val_transforms,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


LAST_CHECKPOINT_PATH = "last_checkpoint.pth"
# BEST_MODEL_PATH = "best_swinunetr_model.pth"
BEST_MODEL_PATH = "best_3dunet_model.pth"
LOG_PATH = "training_log.csv"


def load_medical_volume():
    """
    Descarga y carga el dataset Task01_BrainTumour del Medical Segmentation Decathlon.
    Divide los sujetos etiquetados en entrenamiento y validación siguiendo una proporción 80/20.
    """

    root_dir = "./dataset_brats"
    os.makedirs(root_dir, exist_ok=True)

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

    all_labeled_files = load_decathlon_datalist(
        jsonlist,
        is_segmentation=True,
        data_list_key="training",
        base_dir=datadir,
    )

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
    Comprueba las formas de imagen y etiqueta antes de entrenar.
    Sirve para verificar canales, clases y formato de las máscaras.
    """

    batch = next(iter(train_loader))

    print("\n" + "=" * 50)
    print("[CHECK] Forma de image:", batch["image"].shape)
    print("[CHECK] Forma de label:", batch["label"].shape)
    print("[CHECK] Valores únicos de label:", torch.unique(batch["label"]))
    print("=" * 50 + "\n")


def init_log(log_path, start_epoch):
    """
    Crea el archivo de log solo si no existe o si se empieza desde cero.
    Si se reanuda desde checkpoint, conserva el log anterior.
    """

    if start_epoch == 0 or not os.path.exists(log_path):
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


def append_log(
    log_path,
    epoch,
    train_loss,
    val_mean_dice,
    dice_values,
    epoch_time
):
    """
    Añade una fila al CSV de entrenamiento.
    Guarda loss siempre y Dice solo cuando haya validación.
    """

    if dice_values is None:
        dice_values = ["", "", ""]
    else:
        dice_values = list(dice_values)
        while len(dice_values) < 3:
            dice_values.append("")

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            val_mean_dice if val_mean_dice is not None else "",
            dice_values[0],
            dice_values[1],
            dice_values[2],
            epoch_time,
        ])


def save_best_checkpoint(epoch, metric, filename=BEST_MODEL_PATH):
    """
    Guarda el mejor modelo según Dice medio de validación.
    Solo se sobrescribe si el nuevo Dice es mejor.
    """

    checkpoint = {
        "epoch": int(epoch),
        "best_metric": float(metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint, filename)
    print(f"[INFO] Mejor modelo guardado en época {epoch} con Dice {metric:.4f}")


def save_last_checkpoint(
    epoch,
    best_metric,
    best_metric_epoch,
    filename=LAST_CHECKPOINT_PATH
):
    """
    Guarda el último checkpoint de recuperación.
    Este archivo se sobrescribe en cada época.
    """

    checkpoint = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_metric_epoch": int(best_metric_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint, filename)
    print(f"[INFO] Checkpoint de recuperación guardado en época {epoch}")


def load_last_checkpoint(filename=LAST_CHECKPOINT_PATH):
    """
    Carga el último checkpoint si existe.
    Permite continuar el entrenamiento desde la siguiente época.
    """

    if os.path.exists(filename):
        print(f"[INFO] Cargando checkpoint desde {filename}")

        checkpoint = torch.load(
            filename,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_metric_epoch = int(checkpoint.get("best_metric_epoch", -1))

        print(f"[INFO] Continuando desde época {start_epoch + 1}")
        print(f"[INFO] Mejor Dice anterior: {best_metric:.4f} en época {best_metric_epoch}")

        return start_epoch, best_metric, best_metric_epoch

    print("[INFO] No se encontró checkpoint. Entrenando desde cero.")
    return 0, -1.0, -1

def validate(val_loader):
    """
    Evalúa el modelo sobre el conjunto de validación usando sliding window inference.
    Calcula Dice medio y Dice por clase.
    """

    model.eval()

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="Validación"):
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                )

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs, y=val_labels)

        dice_result, not_nans = dice_metric.aggregate()
        dice_metric.reset()

    dice_per_class = dice_result.cpu().numpy()
    dice_per_class = dice_per_class.reshape(-1)

    mean_dice = dice_per_class.mean()

    return mean_dice, dice_per_class

def load_model_for_prediction(checkpoint_path=BEST_MODEL_PATH):
    """
    Carga el mejor modelo guardado para realizar predicciones.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {checkpoint_path}"
        )

    print(f"[INFO] Cargando modelo para predicción desde: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if "best_metric" in checkpoint:
            print(f"[INFO] Dice del checkpoint: {checkpoint['best_metric']:.4f}")

        if "epoch" in checkpoint:
            print(f"[INFO] Época del checkpoint: {checkpoint['epoch']}")

    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()


def normalize_image_for_display(image_2d):
    """
    Normaliza una imagen 2D para visualizarla correctamente.
    """

    image_2d = image_2d.astype(np.float32)

    p1 = np.percentile(image_2d, 1)
    p99 = np.percentile(image_2d, 99)

    image_2d = np.clip(image_2d, p1, p99)
    image_2d = (image_2d - p1) / (p99 - p1 + 1e-8)

    return image_2d


def choose_best_slice(label_np, pred_np):
    """
    Selecciona automáticamente el corte con mayor presencia tumoral.
    Usa la máscara real como prioridad.
    Si la máscara real está vacía, usa la predicción.
    """

    label_mask = label_np > 0
    slice_scores = label_mask.sum(axis=(0, 1))

    if slice_scores.max() == 0:
        pred_mask = pred_np > 0
        slice_scores = pred_mask.sum(axis=(0, 1))

    if slice_scores.max() == 0:
        return label_np.shape[-1] // 2

    return int(np.argmax(slice_scores))


def save_prediction_figure(
    image_np,
    label_np,
    pred_np,
    case_idx,
    slice_idx,
    output_dir,
    channel=0
):
    """
    Guarda una figura comparando imagen MRI, máscara real,
    predicción y superposición.
    """

    image_slice = image_np[channel, :, :, slice_idx]
    label_slice = label_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]

    image_slice = normalize_image_for_display(image_slice)

    label_masked = np.ma.masked_where(label_slice == 0, label_slice)
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title(f"MRI canal {channel}")
    axes[0].axis("off")

    axes[1].imshow(label_slice, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Máscara real")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="viridis", vmin=0, vmax=3)
    axes[2].set_title("Predicción")
    axes[2].axis("off")

    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(label_masked, cmap="Greens", alpha=0.45, vmin=0, vmax=3)
    axes[3].imshow(pred_masked, cmap="Reds", alpha=0.45, vmin=0, vmax=3)
    axes[3].set_title("Real verde / Predicción roja")
    axes[3].axis("off")

    plt.suptitle(
        f"Caso {case_idx} - Corte {slice_idx}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = output_dir / f"prediccion_caso_{case_idx}_slice_{slice_idx}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] Imagen guardada: {output_path}")


def predict_and_save_examples(
    val_loader,
    checkpoint_path=BEST_MODEL_PATH,
    output_dir="predicciones",
    num_examples=5,
    channel=0
):
    """
    Genera predicciones sobre varios casos de validación y guarda
    imágenes comparativas entre MRI, máscara real y predicción.

    channel:
        0 suele corresponder a una de las modalidades MRI.
        Si la imagen se ve poco clara, probar con channel=1, 2 o 3.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    load_model_for_prediction(checkpoint_path)

    print("\n[INFO] Generando predicciones cualitativas...\n")

    with torch.no_grad():
        for case_idx, val_data in enumerate(val_loader):

            if case_idx >= num_examples:
                break

            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                )

            pred = torch.argmax(val_outputs, dim=1)

            image_np = val_inputs[0].detach().cpu().numpy()
            label_np = val_labels[0, 0].detach().cpu().numpy().astype(np.uint8)
            pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)

            slice_idx = choose_best_slice(label_np, pred_np)

            save_prediction_figure(
                image_np=image_np,
                label_np=label_np,
                pred_np=pred_np,
                case_idx=case_idx,
                slice_idx=slice_idx,
                output_dir=output_dir,
                channel=channel,
            )

            del val_inputs, val_labels, val_outputs, pred

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n[INFO] Predicciones finalizadas.")
    print(f"[INFO] Imágenes guardadas en: {output_dir}")


def train(train_loader, val_loader):
    """
    Entrenamiento principal del modelo.

    Guarda:
    - loss de entrenamiento en cada época;
    - Dice medio y Dice por clase cuando hay validación;
    - último checkpoint en cada época;
    - mejor modelo cuando mejora el Dice de validación.
    """

    print("\nIniciando entrenamiento...\n")

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch, best_metric, best_metric_epoch = load_last_checkpoint()

    init_log(LOG_PATH, start_epoch)

    total_start = time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time()

        print("-" * 50)
        print(f"Época {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        epoch_loss = 0.0
        step_count = 0

        loop_batches = tqdm(
            train_loader,
            desc=f"Entrenamiento {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch_data in loop_batches:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=torch.cuda.is_available()):
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

        val_mean_dice = None
        dice_values = None

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_mean_dice, dice_per_class = validate(val_loader)

            print(f"[VAL] Dice medio: {val_mean_dice:.4f}")
            print(f"[VAL] Dice por clase: {dice_per_class}")

            scheduler.step(val_mean_dice)

            dice_values = list(dice_per_class)

            if val_mean_dice > best_metric:
                best_metric = val_mean_dice
                best_metric_epoch = epoch + 1

                save_best_checkpoint(
                    epoch=epoch + 1,
                    metric=best_metric,
                    filename=BEST_MODEL_PATH,
                )

        append_log(
            log_path=LOG_PATH,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_mean_dice=val_mean_dice,
            dice_values=dice_values,
            epoch_time=epoch_time,
        )

        save_last_checkpoint(
            epoch=epoch + 1,
            best_metric=best_metric,
            best_metric_epoch=best_metric_epoch,
            filename=LAST_CHECKPOINT_PATH,
        )

    total_time = time() - total_start

    print("\nEntrenamiento finalizado.")
    print(f"Mejor Dice medio: {best_metric:.4f} en época {best_metric_epoch}")
    print(f"Tiempo total: {total_time / 3600:.2f} horas")
    print(f"Logs guardados en: {LOG_PATH}")
    print(f"Último checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Mejor modelo: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train_loader, val_loader = load_medical_volume()

    RUN_TRAINING = False
    RUN_PREDICTION = True

    if RUN_TRAINING:
        check_batch_shapes(train_loader)
        train(train_loader, val_loader)

    if RUN_PREDICTION:
        predict_and_save_examples(
            val_loader=val_loader,
            checkpoint_path=BEST_MODEL_PATH,
            output_dir="predicciones",
            num_examples=5,
            channel=0,
        )