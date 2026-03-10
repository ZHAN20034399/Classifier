"""
SulMineralClassifier - 硫化矿物成因类型分类软件包
SulMineralClassifier - Sulfide Mineral Genetic Type Classification Package

支持黄铜矿（Ccp）和黄铁矿（Py）两种矿物，
基于 MLP、RF、XGBoost 和 Stacking 四种机器学习分类器，
实现矿床类型（SEDEX vs VMS）的智能判别。

Supports Chalcopyrite (Ccp) and Pyrite (Py) minerals,
using MLP, RF, XGBoost, and Stacking classifiers
to discriminate deposit types (SEDEX vs VMS).
"""

__version__ = "1.0.0"
__author__ = "SulMineralClassifier"

from . import config
from . import core
from . import models
from . import visualization
from . import explain

__all__ = ["config", "core", "models", "visualization", "explain"]
