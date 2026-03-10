"""
models/mlp_model.py - MLP 多层感知机分类器 / MLP Classifier

构建 Pipeline([StandardScaler, MLPClassifier])，超参数从 config 获取。
Builds Pipeline([StandardScaler, MLPClassifier]) with hyperparameters from config.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..config import get_params


def build_mlp_pipeline(mineral_type: str) -> Pipeline:
    """
    构建 MLP 分类器 Pipeline（StandardScaler + MLPClassifier）。
    Build the MLP classifier pipeline (StandardScaler + MLPClassifier).

    MLP 对特征尺度敏感，因此在 Pipeline 中包含 StandardScaler。
    MLP is sensitive to feature scale, so StandardScaler is included in the pipeline.

    Parameters
    ----------
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿）/ 'ccp' or 'py'

    Returns
    -------
    Pipeline
        未拟合的 sklearn Pipeline / Unfitted sklearn Pipeline
    """
    params = get_params(mineral_type)["mlp"]

    mlp = MLPClassifier(
        activation=params["activation"],          # 激活函数 / Activation function
        solver=params["solver"],                  # 优化器 / Optimizer
        alpha=params["alpha"],                    # L2 正则化项 / L2 regularization
        hidden_layer_sizes=params["hidden_layer_sizes"],  # 隐藏层结构 / Hidden layer sizes
        max_iter=params["max_iter"],              # 最大迭代次数 / Maximum iterations
        random_state=1,                           # 随机种子 / Random seed
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),             # 标准化 / Feature standardization
        ("mlpclassifier", mlp),
    ])

    return pipeline
