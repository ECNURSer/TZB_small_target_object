"""TensorBoard setup for YOLO26 OBB training."""

from ultralytics import settings


def enable_tensorboard_scalars() -> None:
    """Enable official scalar logging but skip unsupported YOLO26 OBB graph tracing."""
    settings.update({"tensorboard": True})
    from ultralytics.utils.callbacks import tensorboard

    tensorboard.callbacks.pop("on_train_start", None)

