"""
visualization/confusion_plot.py - 混淆矩阵可视化 / Confusion Matrix Visualization

绘制含数量和百分比的混淆矩阵热力图，支持保存为 SVG/PDF/JPG 格式。
Plots a confusion matrix heatmap with counts and percentages,
supports saving as SVG / PDF / JPG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as _confusion_matrix

from ..config import CLASS_NAMES


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = CLASS_NAMES,
    model_name: str = "Model",
    save_dir: str = None,
    show: bool = True,
) -> None:
    """
    绘制混淆矩阵热力图（含每格数量与百分比）。
    Plot a confusion matrix heatmap with per-cell counts and percentages.

    参数 / Parameters
    ----------
    y_true : np.ndarray
        真实标签（整数编码）/ True labels (integer-encoded)
    y_pred : np.ndarray
        预测标签（整数编码）/ Predicted labels (integer-encoded)
    class_names : list, optional
        类别名称列表，默认 ['SEDEX', 'VMS'] / Class names, default ['SEDEX', 'VMS']
    model_name : str, optional
        模型名称（用于图题和文件名）/ Model name for title and filename
    save_dir : str, optional
        图片保存目录；为 None 时只显示不保存 / Directory to save figures; None = display only
    show : bool, optional
        是否调用 plt.show() 显示图片，默认 True；无头环境下可设为 False
        Whether to call plt.show(); set to False for headless/automated workflows

    保存格式 / Saved formats
    -----------------------
    SVG, PDF, JPG（dpi=600）
    """
    cm = _confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)

    # 按列归一化为百分比 / Normalize by column to get percentages
    cm_df_pct = cm_df.div(cm_df.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    plt.rc("font", family="Arial", size=8)

    # 绘制热力图（不显示默认数值注释）
    # Draw heatmap without default annotations
    sns.heatmap(
        cm_df,
        linewidths=0.5,
        ax=ax,
        cmap="Blues",
        annot=False,
    )

    # 自定义每格文字（数量 + 百分比）/ Custom text per cell (count + percentage)
    norm = plt.Normalize(vmin=cm_df.values.min(), vmax=cm_df.values.max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)

    for i in range(len(cm_df)):
        for j in range(len(cm_df)):
            value = cm_df.iloc[i, j]
            percentage = cm_df_pct.iloc[i, j]
            # 根据背景色决定文字颜色（深色背景用白字）
            # Choose text color based on background brightness
            rgba = sm.to_rgba(value)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.5 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value}\n{percentage:.1f}%",
                ha="center",
                va="center",
                color=color,
                family="Arial",
                size=8,
            )

    ax.set_title(f"Test set confusion matrix ({model_name})", fontsize=8)
    ax.set_xlabel("Predictions", fontsize=8)
    ax.set_ylabel("True labels", fontsize=8)
    ax.set_xticklabels(class_names, rotation=45, fontsize=8)
    ax.set_yticklabels(class_names, rotation=0, fontsize=8)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, f"confusion_matrix_{model_name}")
        for fmt in ("svg", "pdf", "jpg"):
            path = f"{base}.{fmt}"
            plt.savefig(path, dpi=600, format=fmt)
            print(
                f"[可视化] 混淆矩阵已保存 / [Viz] Confusion matrix saved: {path}"
            )

    if show:
        plt.show()
    plt.close()
