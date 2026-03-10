"""
models/rf_model.py - 随机森林分类器 / Random Forest Classifier

构建含 oob_score 和 class_weight='balanced' 的随机森林 Pipeline。
Builds a Random Forest pipeline with oob_score and class_weight='balanced'.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from ..config import get_params


def build_rf_pipeline(mineral_type: str) -> RandomForestClassifier:
    """
    构建随机森林分类器 Pipeline。
    Build the Random Forest classifier pipeline.

    Parameters
    ----------
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿）/ 'ccp' or 'py'

    Returns
    -------
    sklearn pipeline / RandomForestClassifier wrapped in make_pipeline
        未拟合的分类器 Pipeline / Unfitted classifier pipeline
    """
    params = get_params(mineral_type)["rf"]

    rf = RandomForestClassifier(
        n_estimators=params["n_estimators"],        # 决策树数量 / Number of trees
        max_depth=params["max_depth"],              # 最大深度 / Maximum tree depth
        bootstrap=params["bootstrap"],              # 是否自举采样 / Bootstrap sampling
        min_samples_leaf=params["min_samples_leaf"],  # 叶节点最小样本数 / Min samples per leaf
        min_samples_split=params["min_samples_split"],  # 分裂节点最小样本数 / Min samples to split
        oob_score=True,                             # 袋外误差评估 / Out-of-bag score
        class_weight="balanced",                    # 处理类别不平衡 / Handle class imbalance
        random_state=10,                            # 随机种子 / Random seed
    )

    pipeline = make_pipeline(rf)
    return pipeline
