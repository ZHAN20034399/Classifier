"""
core/__init__.py

导出 core 子包的主要公共接口。
Exports the main public interface of the core sub-package.
"""

from .data_loader import load_training_data, load_prediction_data
from .preprocessing import preprocess_data, split_data, encode_target
from .trainer import train_model, train_all_models
from .predictor import predict, predict_proba, generate_report, save_report

__all__ = [
    "load_training_data",
    "load_prediction_data",
    "preprocess_data",
    "split_data",
    "encode_target",
    "train_model",
    "train_all_models",
    "predict",
    "predict_proba",
    "generate_report",
    "save_report",
]
