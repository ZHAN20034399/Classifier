"""
core/data_loader.py - 数据加载模块 / Data Loading Module

提供从 Excel 文件加载训练数据和待预测数据的函数。
Provides functions to load training data and prediction data from Excel files.
"""

import pandas as pd
from typing import Tuple

from ..config import get_elements, TARGET_COL


def load_training_data(file_path: str, mineral_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载训练数据 Excel 文件，根据矿物类型自动选取对应的特征列。
    Load the training Excel file and select feature columns by mineral type.

    Parameters
    ----------
    file_path : str
        Excel 文件路径 / Path to the Excel training file
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿） / 'ccp' for Chalcopyrite, 'py' for Pyrite

    Returns
    -------
    X : pd.DataFrame
        特征矩阵 / Feature matrix
    y : pd.Series
        目标变量（原始字符串标签）/ Target variable (raw string labels)
    """
    elements = get_elements(mineral_type)
    cols = [TARGET_COL] + elements

    data = pd.read_excel(file_path)

    # 只保留所需列（忽略不存在的列并给出提示）
    # Keep only required columns (warn about missing ones)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"训练文件缺少以下列 / Training file is missing columns: {missing}\n"
            f"文件包含的列 / File columns: {list(data.columns)}"
        )

    df = data[cols]
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def load_prediction_data(file_path: str, mineral_type: str) -> pd.DataFrame:
    """
    加载待预测数据 Excel 文件，根据矿物类型自动选取特征列。
    Load the prediction Excel file and select feature columns by mineral type.

    Parameters
    ----------
    file_path : str
        Excel 文件路径 / Path to the Excel prediction file
    mineral_type : str
        'ccp' 或 'py' / 'ccp' or 'py'

    Returns
    -------
    pd.DataFrame
        仅含特征列的 DataFrame（已将各列转为数值，删除含缺失值的行）
        DataFrame with feature columns only (numeric-coerced, NaN rows dropped)
    """
    elements = get_elements(mineral_type)

    data = pd.read_excel(file_path)

    missing = [e for e in elements if e not in data.columns]
    if missing:
        raise ValueError(
            f"预测文件缺少以下列 / Prediction file is missing columns: {missing}\n"
            f"文件包含的列 / File columns: {list(data.columns)}"
        )

    df = data[elements].copy()
    # 将各元素列强制转换为数值型，无法转换的置为 NaN
    # Coerce each element column to numeric; non-numeric values become NaN
    for col in elements:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna()
    df = df.reset_index(drop=True)
    after = len(df)

    if before > after:
        print(
            f"[数据加载] 已删除 {before - after} 行含缺失值的样本，"
            f"剩余 {after} 个有效样本。\n"
            f"[Data Loader] Dropped {before - after} rows with missing values; "
            f"{after} valid samples remain."
        )

    return df
