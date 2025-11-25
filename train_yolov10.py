import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["MULTIPROCESSING_START_METHOD"] = "spawn"

from ultralytics import YOLO
import torch

def main():

    print("\n==============================")
    print("📁 ENTRENANDO v15 (desde cero, dataset limpio)")
    print("==============================\n")

    # 👉 ENTRENAMOS DESDE CERO PERO CON PESOS BASE YOLO
    model = YOLO("yolov10s.pt")

    model.train(
        data="../../datasets/violencia_estadio_v13/data.yaml",  # 👈 usa el dataset NUEVO Y LIMPIO
        epochs=40,                # 👈 mejor para dataset nuevo
        imgsz=640,                # 👈 excelente para objetos delgados como bats
        batch=8,
        device=0,
        pretrained=True,          # 👈 usa pesos de YOLO (NO tu checkpoint viejo)
        optimizer="SGD",
        lr0=0.0005,
        amp=True,
        patience=20,              # 👈 subimos paciencia para dataset nuevo
        close_mosaic=10,
        workers=2,
        name="violencia_estadio_yolov10s_v15",
        project="runs/train",

        # EXTRA: mejora objetos pequeños/finos
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        erasing=0.4,
        auto_augment="randaugment"
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
