"""
core/predictor.py - 模型预测与结果导出模块 / Model Prediction & Export Module

提供预测、概率获取、报告生成和 Excel 导出功能。
Provides prediction, probability retrieval, report generation, and Excel export.
"""

import numpy as np
import pandas as pd
from collections import Counter

from ..config import CLASS_NAMES


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    使用训练好的模型对特征数据进行类别预测。
    Predict class labels using the fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        已训练的模型 / Fitted model
    X : pd.DataFrame
        待预测特征矩阵 / Feature matrix to predict

    Returns
    -------
    np.ndarray
        整数编码的预测标签数组 / Integer-encoded predicted label array
    """
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    获取各类别的预测概率。
    Get predicted class probabilities.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        已训练的模型 / Fitted model
    X : pd.DataFrame
        待预测特征矩阵 / Feature matrix to predict

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
        各样本的类别概率 / Class probabilities for each sample
    """
    return model.predict_proba(X)


def generate_report(
    model,
    X: pd.DataFrame,
    class_names: list = CLASS_NAMES,
) -> pd.DataFrame:
    """
    生成包含预测结果和概率的 DataFrame 报告。
    Generate a DataFrame report containing predictions and probabilities.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        已训练的模型 / Fitted model
    X : pd.DataFrame
        待预测特征矩阵 / Feature matrix to predict
    class_names : list, optional
        类别名称列表，默认 ['SEDEX', 'VMS'] / Class names, default ['SEDEX', 'VMS']

    Returns
    -------
    pd.DataFrame
        包含 'predicted_type'、'{class}_proba' 列的报告 DataFrame
        Report DataFrame with 'predicted_type' and '{class}_proba' columns
    """
    y_pred_int = predict(model, X)
    y_proba = predict_proba(model, X)

    # 将整数编码转回类别名称
    # Map integer codes back to class names
    y_pred_labels = [class_names[i] for i in y_pred_int]

    # 统计各类别数量并打印摘要
    # Summarize prediction counts
    counter = Counter(y_pred_labels)
    print("\n[预测摘要 / Prediction Summary]")
    print(f"各类别样本数 / Counts per class: {dict(counter)}")
    most_common = counter.most_common(1)[0][0]
    print(f"最多预测类型 / Most predicted type: {most_common}")

    # 构建报告 DataFrame
    # Build report DataFrame
    proba_cols = {f"{cn}_proba": y_proba[:, i] for i, cn in enumerate(class_names)}
    report = pd.DataFrame({"predicted_type": y_pred_labels, **proba_cols})

    return report


def save_report(report_df: pd.DataFrame, output_path: str) -> None:
    """
    将预测结果 DataFrame 保存到 Excel 文件。
    Save the prediction report DataFrame to an Excel file.

    Parameters
    ----------
    report_df : pd.DataFrame
        预测结果 DataFrame / Prediction report DataFrame
    output_path : str
        输出 Excel 文件路径 / Output Excel file path
    """
    report_df.to_excel(output_path, index=False)
    print(f"\n[保存] 预测结果已保存至 / [Save] Prediction results saved to: {output_path}")
