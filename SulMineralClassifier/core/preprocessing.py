"""
core/preprocessing.py - 数据预处理模块 / Data Preprocessing Module

提供缺失值处理、目标变量编码和数据集划分功能。
Provides missing-value handling, target encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split as _split

from ..config import RANDOM_STATE, TEST_SIZE


def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    处理特征矩阵和目标变量中的缺失值。
    Handle missing values in the feature matrix and target variable.

    处理顺序 / Processing order:
    1. 删除目标变量为 NaN 的行 / Drop rows where target is NaN
    2. 删除特征矩阵中含 NaN 的行 / Drop rows where any feature is NaN

    Parameters
    ----------
    X : pd.DataFrame
        原始特征矩阵 / Raw feature matrix
    y : pd.Series
        原始目标变量 / Raw target variable

    Returns
    -------
    X_clean : pd.DataFrame
        清洗后的特征矩阵 / Cleaned feature matrix
    y_clean : pd.Series
        对应的目标变量 / Corresponding target variable
    """
    # 1. 处理目标变量缺失值 / Handle target NaN
    if y.isnull().any():
        y = y.dropna()
        X = X.loc[y.index]

    # 2. 处理特征缺失值 / Handle feature NaN
    X = X.copy()
    # 将各列转为数值型（如有非数值字符串）
    # Coerce all columns to numeric (handle any non-numeric strings)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isnull().any().any():
        X = X.dropna()
        y = y.loc[X.index]

    print(
        f"[预处理] 清洗后样本数：{len(X)}\n"
        f"[Preprocessing] Samples after cleaning: {len(X)}"
    )
    print(
        f"[预处理] 类别分布 / Class distribution:\n{y.value_counts()}"
    )

    return X, y


def encode_target(y: pd.Series) -> Tuple[np.ndarray, pd.Index]:
    """
    对目标变量进行编码（使用 pd.factorize 按字母顺序）。
    Encode the target variable using pd.factorize (alphabetical order → 0=SEDEX, 1=VMS).

    Parameters
    ----------
    y : pd.Series
        原始字符串标签序列 / Raw string label series

    Returns
    -------
    y_encoded : np.ndarray
        整数编码后的目标数组 / Integer-encoded target array
    categories : pd.Index
        编码顺序对应的类别名称 / Category names in encoding order
    """
    y_encoded, categories = pd.factorize(y, sort=True)
    return y_encoded, categories


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    将数据集划分为训练集和测试集。
    Split the dataset into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵 / Feature matrix
    y : np.ndarray
        编码后的目标数组 / Encoded target array
    test_size : float, optional
        测试集比例，默认 0.2 / Test proportion, default 0.2
    random_state : int, optional
        随机种子，默认 2 / Random seed, default 2

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = _split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"[划分] 训练集：{len(X_train)} 条，测试集：{len(X_test)} 条\n"
        f"[Split] Train: {len(X_train)} samples, Test: {len(X_test)} samples"
    )
    return X_train, X_test, y_train, y_test
