"""
config.py - 超参数配置 / Hyperparameter Configuration

定义特征列、模型超参数、类别名称和通用训练参数。
Defines feature columns, model hyperparameters, class names, and general training parameters.
"""

# ── 特征元素列 / Feature element columns ─────────────────────────────────────

# 黄铜矿 (Chalcopyrite/Ccp) 使用的 8 个特征元素
# 8 feature elements for Chalcopyrite (Ccp)
CCP_ELEMENTS = ["Co", "Ni", "Zn", "Cd", "Sb", "Pb", "Ag", "Se"]

# 黄铁矿 (Pyrite/Py) 使用的 10 个特征元素
# 10 feature elements for Pyrite (Py)
PY_ELEMENTS = ["Co", "Ni", "Zn", "Cu", "Sb", "Pb", "Ag", "Se", "As", "Bi"]

# ── 目标列与分类名称 / Target column and class names ─────────────────────────

TARGET_COL = "Deposit type"

# 二分类目标：SEDEX vs VMS（按字母顺序 factorize 后 0=SEDEX, 1=VMS）
# Binary classification target: SEDEX vs VMS (0=SEDEX, 1=VMS after alphabetic factorize)
CLASS_NAMES = ["SEDEX", "VMS"]

# ── 通用训练参数 / General training parameters ───────────────────────────────

RANDOM_STATE = 2          # 随机种子 / Random seed
TEST_SIZE = 0.2           # 测试集比例 / Test set proportion
CV_FOLDS = 5              # 交叉验证折数 / Cross-validation folds

# ── 各模型最优超参数 / Optimal hyperparameters per model ─────────────────────
#   参数来源：论文 Table 3 / Source: Paper Table 3

CCP_PARAMS = {
    "rf": {
        "n_estimators": 130,
        "max_depth": 11,
        "bootstrap": True,
        "min_samples_leaf": 3,
        "min_samples_split": 5,
    },
    "xgb": {
        "max_depth": 3,
        "learning_rate": 0.2,
        "n_estimators": 50,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    },
    "mlp": {
        "activation": "tanh",
        "solver": "adam",
        "alpha": 0.0001,
        "hidden_layer_sizes": (50, 50),
        "max_iter": 200,
    },
}

PY_PARAMS = {
    "rf": {
        "n_estimators": 130,
        "max_depth": 11,
        "bootstrap": True,
        "min_samples_leaf": 3,
        "min_samples_split": 5,
    },
    "xgb": {
        "max_depth": 9,
        "learning_rate": 0.2,
        "n_estimators": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "mlp": {
        "activation": "tanh",
        "solver": "adam",
        "alpha": 0.0001,
        "hidden_layer_sizes": (50, 50),
        "max_iter": 200,
    },
}


def get_params(mineral_type: str) -> dict:
    """
    根据矿物类型返回对应的超参数字典。
    Return the hyperparameter dictionary for the given mineral type.

    Parameters
    ----------
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿） / 'ccp' or 'py'

    Returns
    -------
    dict
        包含 'rf', 'xgb', 'mlp' 键的超参数字典
        Hyperparameter dict with keys 'rf', 'xgb', 'mlp'
    """
    if mineral_type.lower() == "ccp":
        return CCP_PARAMS
    elif mineral_type.lower() == "py":
        return PY_PARAMS
    else:
        raise ValueError(f"Unknown mineral_type '{mineral_type}'. Must be 'ccp' or 'py'.")


def get_elements(mineral_type: str) -> list:
    """
    根据矿物类型返回对应的特征元素列表。
    Return the feature element list for the given mineral type.

    Parameters
    ----------
    mineral_type : str
        'ccp' 或 'py' / 'ccp' or 'py'

    Returns
    -------
    list[str]
        特征元素名称列表 / List of feature element names
    """
    if mineral_type.lower() == "ccp":
        return CCP_ELEMENTS
    elif mineral_type.lower() == "py":
        return PY_ELEMENTS
    else:
        raise ValueError(f"Unknown mineral_type '{mineral_type}'. Must be 'ccp' or 'py'.")
