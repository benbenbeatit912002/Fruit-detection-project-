"""Adapter for Ultralytics YOLO11n (Ultralytics, AGPL-3.0)."""

from ultralytics.models.yolo.detect import DetectionTrainer

from .prepare import CLASSES, CONVERTED, PROJECT, save_json, sha
from .train import BATCH, EPOCHS, SEED, SIZE, Run


class ComparisonTrainer(DetectionTrainer):
    def validate(self):
        values = self.run.epoch(self.ema.ema, self.epoch + 1, self.tloss.sum().item(), list(self.lr.values()))
        self.best_fitness = self.run.best
        return {k: values[k] for k in ("map", "map50", "map75")}, values["map"]

    def save_model(self):
        # Run.epoch saves the selected full-precision EMA state, including ties policy.
        pass

    def final_eval(self):
        self.run.finish(self.ema.ema)


def record_training_setup(trainer):
    if list(trainer.data["names"].values()) != list(CLASSES):
        raise ValueError("YOLO class names differ from the evaluation class order")
    if not trainer.amp or trainer.accumulate != 1 or trainer.device.type != "cuda":
        raise RuntimeError("Trainer startup differs from the recorded GPU protocol")
    if len(trainer.train_loader.dataset) != 218:
        raise RuntimeError("Ultralytics dropped training images")
    labels = trainer.train_loader.dataset.labels
    if sum(len(record["cls"]) for record in labels) != 1603:
        raise RuntimeError("Ultralytics dropped training labels")
    config = trainer.run.config
    config["effective_trainer_arguments"] = {
        k: v for k, v in vars(trainer.args).items()
        if k not in {"data", "model", "project", "save_dir", "name"}
    }
    config["actual_optimizer"] = type(trainer.optimizer).__name__
    config["gradient_clip_norm"] = 10.0
    save_json(trainer.run.directory / "config.json", config)
    trainer.run.start()


def train_yolo(name, directory, audit):
    pretrained = name == "yolo11n-coco"
    weights = PROJECT / "weights" / "yolo11n.pt"
    if not weights.is_file():
        raise FileNotFoundError("Provide the official weights/yolo11n.pt for transfer and the upstream AMP probe.")
    digest = sha(weights)
    if digest != "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1":
        raise ValueError("Unexpected YOLO11n checkpoint; see the pinned release URL")
    config = dict(initialization="COCO" if pretrained else "random", optimizer="AdamW", lr0=.001,
                  initial_weights_sha256=digest if pretrained else None, architecture="Ultralytics yolo11n.yaml",
                  evaluation_weights="exponential moving average (FP32)",
                  preprocessing="letterbox RGB / 255; inverse scale and padding for original pixels")
    run = Run(name, directory, audit, config)
    trainer = ComparisonTrainer(overrides={
        "model": str(weights) if pretrained else "yolo11n.yaml", "pretrained": pretrained,
        "data": str((CONVERTED / "data.yaml").resolve()), "project": str(directory.resolve()),
        "name": "ultralytics", "device": "0", "epochs": EPOCHS, "seed": SEED, "imgsz": SIZE,
        "batch": BATCH, "nbs": BATCH, "workers": 0, "optimizer": "AdamW", "lr0": .001,
        "amp": True, "deterministic": True, "cache": False, "val": True, "plots": False,
        "conf": .05, "iou": .70, "max_det": 100, "save": False, "verbose": False, "patience": 0,
    })
    trainer.run = run
    trainer.add_callback("on_train_start", record_training_setup)
    trainer.train()
