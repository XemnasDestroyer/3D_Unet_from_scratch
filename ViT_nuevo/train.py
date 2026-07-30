import os
import random
import json
import csv
from time import time
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers import sliding_window_inference

# =========================================================================
# IMPORTS DESDE TU CONFIGURACIÓN (config.py)
# =========================================================================
from config import (
    NUM_EPOCHS,
    VAL_INTERVAL,
    BATCH_SIZE,
    OUT_CHANNELS,
    ROI_SIZE,
    MODEL_NAME,
    device,
    train_transforms,
    val_transforms,
    create_model,
    optimizer as base_optimizer,
    loss_function,
    dice_metric,
    post_pred,
    post_label,
    scheduler as base_scheduler,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

NUM_RUNS = 3

# Configuración de semillas
SEEDS = [42, 123, 999]  # Semillas para las 3 ejecuciones
SPLIT_SEED = 42         # Semilla fija para mantener la partición 80/10/10

# =========================================================================
# 1. GESTIÓN DEL DATASET Y PARTICIÓN PERSISTENTE (80/10/10)
# =========================================================================
def load_medical_volume():
    """Carga el dataset y fija la partición 80/10/10 guardada en JSON."""
    root_dir = "./dataset_brats"
    split_json_path = os.path.join(root_dir, "split_dataset.json")
    os.makedirs(root_dir, exist_ok=True)

    # Descarga e inicialización Decathlon
    DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        download=True,
        cache_num=5,
    )

    list_dir = os.path.join(root_dir, "Task01_BrainTumour")
    jsonlist = os.path.join(list_dir, "dataset.json")

    if os.path.exists(split_json_path):
        print(f"\n[INFO] Cargando partición existente desde {split_json_path}...")
        with open(split_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        train_files, val_files, test_files = splits["train"], splits["val"], splits["test"]
    else:
        print("\n[INFO] Generando nueva partición 80/10/10...")
        all_labeled_files = load_decathlon_datalist(
            jsonlist, is_segmentation=True, data_list_key="training", base_dir=list_dir
        )

        random.seed(SPLIT_SEED)
        random.shuffle(all_labeled_files)

        total_files = len(all_labeled_files)
        train_end = int(total_files * 0.8)
        val_end = train_end + int(total_files * 0.1)

        train_files = all_labeled_files[:train_end]
        val_files = all_labeled_files[train_end:val_end]
        test_files = all_labeled_files[val_end:]

        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump({"train": train_files, "val": val_files, "test": test_files}, f, indent=4)

    print("=" * 50)
    print(f"[INFO] Entrenamiento (80%): {len(train_files)} sujetos")
    print(f"[INFO] Validación    (10%): {len(val_files)} sujetos")
    print(f"[INFO] Test          (10%): {len(test_files)} sujetos")
    print("=" * 50 + "\n")

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.2, num_workers=0)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.2, num_workers=0)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# =========================================================================
# 2. FUNCIONES AUXILIARES (REPRODUCIBILIDAD, CSV Y VALIDACIÓN)
# =========================================================================
def set_seed(seed):
    """Fija la semilla para la inicialización reproducible del modelo."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_csv_log(filepath, num_classes=3):
    """Inicializa la cabecera del CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = ["epoch", "train_loss", "val_mean_dice"]
    for c in range(num_classes):
        header.append(f"val_dice_class_{c}")
    header.append("epoch_time_sec")

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_log(filepath, epoch, train_loss, val_mean_dice, dice_per_class, epoch_time):
    """Registra la época actual en el archivo CSV."""
    val_dice_str = f"{val_mean_dice:.6f}" if val_mean_dice is not None else ""
    row = [epoch, f"{train_loss:.6f}", val_dice_str]

    # Número de clases a registrar (excluyendo el fondo si OUT_CHANNELS=4)
    num_eval_classes = OUT_CHANNELS - 1 if OUT_CHANNELS > 1 else 1

    if dice_per_class is not None:
        for dice_c in dice_per_class:
            row.append(f"{dice_c:.6f}")
    else:
        row.extend([""] * num_eval_classes)

    row.append(f"{epoch_time:.2f}")

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def validate(model, val_loader):
    """Función de validación utilizando inferencia por ventana deslizante."""
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

            # Inferencia 3D dividida en sub-bloques ROI_SIZE
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=ROI_SIZE,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )

            # Post-procesamiento
            val_outputs = [post_pred(i) for i in val_outputs]
            val_labels = [post_label(i) for i in val_labels]

            # Actualizar métrica Dice
            dice_metric(y_pred=val_outputs, y=val_labels)

        res = dice_metric.aggregate()

        # Si get_not_nans=True en config.py, aggregate() devuelve una tupla (metric_tensor, not_nans)
        if isinstance(res, (list, tuple)):
            metric_batch = res[0]
        else:
            metric_batch = res

        val_mean_dice = metric_batch.mean().item()
        dice_per_class = [m.item() for m in metric_batch]

    return val_mean_dice, dice_per_class


# =========================================================================
# 3. ENTRENAMIENTO DE UN RUN INDIVIDUAL (100 ÉPOCAS)
# =========================================================================
def train_single_run(run_idx, seed, train_loader, val_loader):
    print("\n" + "=" * 60)
    print(f"🚀 INICIANDO RUN {run_idx + 1}/3 (Modelo: {MODEL_NAME.upper()} | Semilla: {seed})")
    print("=" * 60 + "\n")

    set_seed(seed)

    # Instanciamos el modelo limpio para esta run desde la función de config.py
    current_model = create_model(MODEL_NAME).to(device)

    # Optimizador y Scheduler independientes por Run
    optimizer = torch.optim.AdamW(
        current_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # Carpetas y CSV de esta ejecución
    run_dir = f"./resultados/{MODEL_NAME}/run_{run_idx + 1}"
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    csv_log_path = os.path.join(run_dir, f"metrics_run_{run_idx + 1}.csv")

    num_eval_classes = OUT_CHANNELS - 1 if OUT_CHANNELS > 1 else 1
    init_csv_log(csv_log_path, num_classes=num_eval_classes)

    best_metric = -1.0
    best_metric_epoch = -1

    for epoch in range(NUM_EPOCHS):
        epoch_start = time()

        current_model.train()
        epoch_loss = 0.0
        step_count = 0

        loop_batches = tqdm(
            train_loader,
            desc=f"[{MODEL_NAME.upper()} - Run {run_idx + 1}] Época {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch_data in loop_batches:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            # Si el parche tiene formato (B, N, C, H, W, D) por RandCropByPosNegLabeld, aplanamos
            if inputs.ndim == 6:
                b, n, c, h, w, d = inputs.shape
                inputs = inputs.view(b * n, c, h, w, d)
                labels = labels.view(b * n, -1, h, w, d)

            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = current_model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1
            loop_batches.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / step_count if step_count > 0 else 0.0
        epoch_time = time() - epoch_start

        val_mean_dice = None
        dice_per_class = None

        # Validación
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_mean_dice, dice_per_class = validate(current_model, val_loader)
            scheduler.step(val_mean_dice)

            print(f"\n[RUN {run_idx + 1} | ÉPOCA {epoch + 1}] Train Loss: {train_loss:.4f} | Val Mean Dice: {val_mean_dice:.4f}")

            if val_mean_dice > best_metric:
                best_metric = val_mean_dice
                best_metric_epoch = epoch + 1
                torch.save(
                    {"state_dict": current_model.state_dict(), "metric": best_metric, "epoch": best_metric_epoch},
                    best_model_path,
                )
                print(f"⭐ ¡Nuevo mejor modelo guardado en: {best_model_path}!")

        # Registrar época en CSV
        append_csv_log(
            filepath=csv_log_path,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_mean_dice=val_mean_dice,
            dice_per_class=dice_per_class,
            epoch_time=epoch_time,
        )

    print(f"\n✅ Run {run_idx + 1} finalizado. Mejor Dice: {best_metric:.4f} (Época {best_metric_epoch})")
    return best_metric, csv_log_path


# =========================================================================
# 4. CÁLCULO DEL RESUMEN GLOBAL (MEDIAS Y DESVIACIÓN TÍPICA)
# =========================================================================
def generar_resumen_csv(csv_paths):
    import pandas as pd

    dfs = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(dfs)

    summary = combined.groupby("epoch").agg({
        "train_loss": ["mean", "std"],
        "val_mean_dice": ["mean", "std"]
    })

    summary.columns = [
        "train_loss_mean", "train_loss_std",
        "val_mean_dice_mean", "val_mean_dice_std"
    ]

    out_csv = f"./resultados/{MODEL_NAME}/resumen_promedio_3_runs.csv"
    summary.to_csv(out_csv)
    print("\n" + "=" * 60)
    print(f"📊 RESUMEN GENERAL GUARDADO EN: {out_csv}")
    print("=" * 60)


# =========================================================================
# 5. EXECUTION MAIN
# =========================================================================
def main():
    train_loader, val_loader = load_medical_volume()

    resultados_metrics = []
    csv_paths = []

    for run_idx in range(3):
        seed_actual = SEEDS[run_idx]
        best_metric, csv_path = train_single_run(
            run_idx=run_idx,
            seed=seed_actual,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        resultados_metrics.append(best_metric)
        csv_paths.append(csv_path)

    generar_resumen_csv(csv_paths)

    mean_dice = np.mean(resultados_metrics)
    std_dice = np.std(resultados_metrics)

    print("\n" + "=" * 60)
    print(f"🏆 RESULTADOS FINALES DE VALIDACIÓN PARA [{MODEL_NAME.upper()}]")
    print("=" * 60)
    for i, dice in enumerate(resultados_metrics):
        print(f"Run {i + 1}: Mejor Dice = {dice:.4f}")
    print(f"\n🎯 Media Global: {mean_dice:.4f} ± {std_dice:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()