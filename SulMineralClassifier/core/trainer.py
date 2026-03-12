"""
core/trainer.py - 模型训练模块 / Model Training Module

提供单模型训练和全模型训练的统一入口，打印训练集/测试集评估报告。
Unified entry point for single-model and all-model training,
printing train/test evaluation reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ..config import CLASS_NAMES
from ..models.mlp_model import build_mlp_pipeline
from ..models.rf_model import build_rf_pipeline
from ..models.xgb_model import build_xgb_pipeline
from ..models.stacking_model import build_stacking_model

# 支持的模型名称映射 / Supported model name mapping
MODEL_BUILDERS = {
    "mlp": build_mlp_pipeline,
    "rf": build_rf_pipeline,
    "xgb": build_xgb_pipeline,
    "stacking": build_stacking_model,
}


def _print_report(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_names: list = CLASS_NAMES,
) -> None:
    """
    打印训练集和测试集的分类报告与混淆矩阵。
    Print classification report and confusion matrix for train and test sets.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"模型 / Model: {model_name.upper()}")
    print(f"{'='*60}")

    print("\n── 训练集报告 / Training Set Report ──")
    print(classification_report(y_train, y_train_pred, target_names=class_names))
    print("混淆矩阵 / Confusion Matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))

    print("\n── 测试集报告 / Test Set Report ──")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("混淆矩阵 / Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    mineral_type: str,
) -> object:
    """
    根据模型名称构建并训练对应的分类器，打印评估报告。
    Build and train a classifier by name, then print the evaluation report.

    Parameters
    ----------
    model_name : str
        模型名称：'mlp', 'rf', 'xgb', 'stacking' / Model name
    X_train : pd.DataFrame
        训练集特征 / Training features
    X_test : pd.DataFrame
        测试集特征 / Test features
    y_train : np.ndarray
        训练集标签 / Training labels
    y_test : np.ndarray
        测试集标签 / Test labels
    mineral_type : str
        'ccp' 或 'py' / 'ccp' or 'py'

    Returns
    -------
    model : fitted sklearn-compatible pipeline/estimator
    """
    name = model_name.lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"不支持的模型名称 / Unsupported model name: '{model_name}'. "
            f"可选 / Choose from: {list(MODEL_BUILDERS.keys())}"
        )

    print(f"\n[训练] 正在训练 {name.upper()} 模型... / [Train] Training {name.upper()} model...")
    model = MODEL_BUILDERS[name](mineral_type)
    model.fit(X_train, y_train)

    _print_report(model, X_train, X_test, y_train, y_test, name)
    return model


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    mineral_type: str,
) -> dict:
    """
    训练全部支持的模型（MLP、RF、XGBoost、Stacking），返回模型字典。
    Train all supported models and return a dict of fitted models.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        训练集和测试集特征 / Train and test features
    y_train, y_test : np.ndarray
        训练集和测试集标签 / Train and test labels
    mineral_type : str
        'ccp' 或 'py' / 'ccp' or 'py'

    Returns
    -------
    dict
        键为模型名称，值为已训练的模型 / {name: fitted_model}
    """
    models = {}
    for name in MODEL_BUILDERS:
        models[name] = train_model(name, X_train, X_test, y_train, y_test, mineral_type)
    return models
