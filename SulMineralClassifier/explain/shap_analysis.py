"""
explain/shap_analysis.py - SHAP 可解释性分析 / SHAP Explainability Analysis

使用 KernelExplainer 计算 SHAP 值并绘制 beeswarm 图。
Computes SHAP values with KernelExplainer and plots beeswarm figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from ..config import CLASS_NAMES


def run_shap_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: list = CLASS_NAMES,
    model_name: str = "Model",
    save_dir: str = None,
) -> None:
    """
    对模型执行 SHAP 可解释性分析，绘制各类别的 beeswarm 图。
    Run SHAP analysis on the model and plot per-class beeswarm plots.

    说明 / Notes
    -----------
    - 使用 shap.KernelExplainer，适用于任意黑盒模型（含 Pipeline/Stacking）。
      Uses shap.KernelExplainer, which works for any black-box model
      (including Pipeline and StackingClassifier).
    - KernelExplainer 计算较慢；若训练集较大，建议传入子集以加速。
      KernelExplainer is slow; pass a subset of X_train for large datasets.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        已训练的模型 / Fitted model
    X_train : pd.DataFrame
        训练集特征（用于 KernelExplainer 背景数据）
        Training features (background data for KernelExplainer)
    X_test : pd.DataFrame
        测试集特征（用于计算 SHAP 值）
        Test features (data for which SHAP values are computed)
    class_names : list, optional
        类别名称列表，默认 ['SEDEX', 'VMS'] / Class names, default ['SEDEX', 'VMS']
    model_name : str, optional
        模型名称（用于图题和文件名）/ Model name for title and filename
    save_dir : str, optional
        图片保存目录；None 时只显示 / Directory to save figures; None = display only
    """
    print(
        f"\n[SHAP] 正在计算 {model_name} 的 SHAP 值，请稍候...\n"
        f"[SHAP] Computing SHAP values for {model_name}, please wait..."
    )

    # 定义预测函数（返回概率矩阵）/ Define predict function (returns probability matrix)
    def predict_fn(X_array: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X_array, columns=X_train.columns)
        return model.predict_proba(df)

    # 限制背景数据量以加速 KernelExplainer（最多取 100 个样本）
    # Limit background data size to speed up KernelExplainer (at most 100 samples)
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)

    # 使用背景数据初始化 KernelExplainer
    # Initialize KernelExplainer with background data
    explainer = shap.KernelExplainer(predict_fn, background)

    # 计算测试集的 SHAP 值 / Compute SHAP values for test set
    shap_values = explainer.shap_values(X_test)

    # 为每个类别绘制 beeswarm 图 / Plot beeswarm for each class
    for i, class_name in enumerate(class_names):
        # 获取该类别的 SHAP 值 / Get SHAP values for this class
        values_i = shap_values[i] if isinstance(shap_values, list) else shap_values

        # 构建 Explanation 对象 / Build Explanation object
        expected_val = (
            explainer.expected_value[i]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
        shap_exp = shap.Explanation(
            base_values=expected_val,
            values=values_i,
            data=X_test.values,
            feature_names=list(X_test.columns),
        )

        # 绘制 beeswarm 图 / Draw beeswarm plot
        plt.figure()
        shap.plots.beeswarm(shap_exp, max_display=10, show=False)
        plt.title(f"SHAP Beeswarm — {model_name} ({class_name})", fontsize=9)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"SHAP_beeswarm_{model_name}_{class_name}.jpg")
            plt.savefig(path, dpi=600, format="jpg", bbox_inches="tight")
            print(
                f"[SHAP] 已保存 / Saved: {path}"
            )

        plt.show()
        plt.close()

    print(f"[SHAP] {model_name} 分析完成 / Analysis complete.")
