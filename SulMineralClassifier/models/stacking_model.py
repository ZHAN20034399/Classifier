"""
models/stacking_model.py - Stacking 集成分类器 / Stacking Ensemble Classifier

基学习器：MLP + RF + XGB
元学习器：Pipeline([StandardScaler, MLPClassifier(hidden_layer_sizes=(50,), max_iter=200)])

Base learners: MLP + RF + XGB
Meta learner: Pipeline([StandardScaler, MLPClassifier(hidden_layer_sizes=(50,), max_iter=200)])
"""

from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .mlp_model import build_mlp_pipeline
from .rf_model import build_rf_pipeline
from .xgb_model import build_xgb_pipeline
from ..config import CV_FOLDS


def build_stacking_model(mineral_type: str) -> StackingClassifier:
    """
    构建 Stacking 集成分类器。
    Build the Stacking ensemble classifier.

    架构 / Architecture:
    - 基学习器 / Base learners: MLP, RF, XGBoost
    - 元学习器 / Meta learner: Pipeline([StandardScaler, MLP(hidden_layer_sizes=(50,))])
    - 交叉验证折数 / CV folds: 5

    Parameters
    ----------
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿）/ 'ccp' or 'py'

    Returns
    -------
    StackingClassifier
        未拟合的 Stacking 集成分类器 / Unfitted Stacking ensemble classifier
    """
    # 基学习器列表 / Define base learners
    estimators = [
        ("mlp", build_mlp_pipeline(mineral_type)),
        ("rf", build_rf_pipeline(mineral_type)),
        ("xgb", build_xgb_pipeline(mineral_type)),
    ]

    # 元学习器：StandardScaler + MLP
    # Meta learner: StandardScaler + MLP
    meta_learner = Pipeline([
        ("scaler", StandardScaler()),
        ("mlpclassifier", MLPClassifier(
            hidden_layer_sizes=(50,),    # 单隐藏层 50 个神经元 / Single hidden layer with 50 units
            max_iter=200,               # 最大迭代次数 / Maximum iterations
            random_state=1,             # 随机种子 / Random seed
        )),
    ])

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=CV_FOLDS,                   # 5 折交叉验证生成元特征 / 5-fold CV for meta-features
        passthrough=False,             # 不传递原始特征给元学习器 / Do not pass raw features to meta-learner
        n_jobs=-1,                     # 并行训练 / Parallel training
    )

    return stacking
