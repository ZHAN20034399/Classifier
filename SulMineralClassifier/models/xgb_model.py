"""
models/xgb_model.py - XGBoost 分类器 / XGBoost Classifier

构建 XGBClassifier Pipeline，含 eval_metric='mlogloss'。
Builds XGBClassifier pipeline with eval_metric='mlogloss'.
"""

from sklearn.pipeline import make_pipeline
import xgboost as xgb

from ..config import get_params


def build_xgb_pipeline(mineral_type: str):
    """
    构建 XGBoost 分类器 Pipeline。
    Build the XGBoost classifier pipeline.

    Parameters
    ----------
    mineral_type : str
        'ccp'（黄铜矿）或 'py'（黄铁矿）/ 'ccp' or 'py'

    Returns
    -------
    sklearn pipeline
        未拟合的 XGBoost Pipeline / Unfitted XGBoost pipeline
    """
    params = get_params(mineral_type)["xgb"]

    xgb_clf = xgb.XGBClassifier(
        max_depth=params["max_depth"],              # 最大树深度 / Maximum tree depth
        learning_rate=params["learning_rate"],      # 学习率 / Learning rate
        n_estimators=params["n_estimators"],        # 弱学习器数量 / Number of estimators
        subsample=params["subsample"],              # 行采样比例 / Row subsampling ratio
        colsample_bytree=params["colsample_bytree"],  # 列采样比例 / Column subsampling ratio
        eval_metric="mlogloss",                     # 评估指标 / Evaluation metric
        use_label_encoder=False,                    # 关闭标签编码警告 / Suppress label encoder warning
        random_state=42,                            # 随机种子 / Random seed
    )

    pipeline = make_pipeline(xgb_clf)
    return pipeline
