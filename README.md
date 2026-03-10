# Classifier

Original Jupyter Notebooks (MLP, RF, XGBoost classifiers for Ccp and Pyrite datasets) are preserved in the root directory.

A fully modular Python package **SulMineralClassifier** has been built on top of these original scripts. See below for details.

---

# SulMineralClassifier

**硫化矿物成因类型分类软件**  
**Sulfide Mineral Genetic Type Classification Software**

基于机器学习的矿床类型智能判别工具，支持黄铜矿（Ccp）和黄铁矿（Py）两种矿物，
实现 **SEDEX** 与 **VMS** 矿床类型的自动分类。

A machine learning–based tool for discriminating sulfide mineral deposit types
(**SEDEX** vs **VMS**), supporting Chalcopyrite (Ccp) and Pyrite (Py).

---

## 功能特点 / Features

- 支持 **MLP**、**RF（随机森林）**、**XGBoost** 和 **Stacking 集成** 四种分类器
- 交互式命令行界面，引导用户完成数据加载→训练→预测全流程
- 自动选择对应矿物的特征元素列
- 混淆矩阵可视化（含数量与百分比），支持 SVG / PDF / JPG 导出
- SHAP 可解释性分析（beeswarm 图）
- 预测结果一键导出为 Excel

---

## 目录结构 / Directory Structure

```
SulMineralClassifier/
├── main.py                    # 交互式主入口 / Interactive entry point
├── config.py                  # 超参数与特征配置 / Hyperparameter & feature config
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载 / Data loading
│   ├── preprocessing.py       # 数据预处理 / Data preprocessing
│   ├── predictor.py           # 预测与结果导出 / Prediction & export
│   └── trainer.py             # 模型训练入口 / Model training entry
│
├── models/
│   ├── __init__.py
│   ├── mlp_model.py           # MLP 分类器 / MLP classifier
│   ├── rf_model.py            # 随机森林分类器 / RF classifier
│   ├── xgb_model.py           # XGBoost 分类器 / XGBoost classifier
│   └── stacking_model.py      # Stacking 集成分类器 / Stacking ensemble
│
├── visualization/
│   ├── __init__.py
│   └── confusion_plot.py      # 混淆矩阵可视化 / Confusion matrix visualization
│
└── explain/
    ├── __init__.py
    └── shap_analysis.py       # SHAP 可解释性分析 / SHAP analysis
```

---

## 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 快速开始 / Quick Start

### 方式一：从上级目录作为模块运行 / Run as a module from the parent directory

```bash
python -m SulMineralClassifier.main
```

### 方式二：从 `SulMineralClassifier/` 目录运行 / Run from inside the package directory

```bash
cd SulMineralClassifier
python main.py
```

---

## 交互流程 / Interactive Workflow

程序启动后，将逐步引导用户完成以下操作：

1. **选择矿物类型**：黄铜矿（Ccp）或黄铁矿（Py）
2. **加载训练数据**：输入 Excel 文件路径（需包含 `Deposit type` 列和对应特征列）
3. **选择分类器**：MLP / RF / XGBoost / Stacking / 全部训练
4. **查看评估报告**：自动打印训练集与测试集的分类报告和混淆矩阵
5. **可视化**：绘制并可选保存混淆矩阵热力图
6. **SHAP 分析**（可选）：计算并绘制各类别的特征重要性 beeswarm 图
7. **预测新数据**（可选）：加载待预测 Excel 文件，输出预测类型与概率
8. **保存结果**（可选）：将预测结果导出为 Excel 文件

---

## 数据格式要求 / Data Format Requirements

### 训练数据 / Training Data

Excel 文件需包含以下列：

| 矿物类型 | 必须包含的列 |
|---|---|
| 黄铜矿 Ccp | `Deposit type`, `Co`, `Ni`, `Zn`, `Cd`, `Sb`, `Pb`, `Ag`, `Se` |
| 黄铁矿 Py | `Deposit type`, `Co`, `Ni`, `Zn`, `Cu`, `Sb`, `Pb`, `Ag`, `Se`, `As`, `Bi` |

`Deposit type` 列的值应为 `SEDEX` 或 `VMS`。

### 预测数据 / Prediction Data

Excel 文件包含对应矿物类型的元素列（无需 `Deposit type` 列）。

---

## 超参数配置 / Hyperparameter Configuration

根据论文 Table 3 设置的最优超参数（见 `config.py`）：

### RF（随机森林）
| 参数 | Ccp | Py |
|---|---|---|
| n_estimators | 130 | 130 |
| max_depth | 11 | 11 |
| min_samples_leaf | 3 | 3 |
| min_samples_split | 5 | 5 |

### XGBoost
| 参数 | Ccp | Py |
|---|---|---|
| max_depth | 3 | 9 |
| learning_rate | 0.2 | 0.2 |
| n_estimators | 50 | 50 |
| colsample_bytree | 1.0 | 0.8 |

### MLP（多层感知机）
| 参数 | Ccp | Py |
|---|---|---|
| activation | tanh | tanh |
| solver | adam | adam |
| hidden_layer_sizes | (50, 50) | (50, 50) |
| alpha | 0.0001 | 0.0001 |

---

## 分类目标 / Classification Target

- **SEDEX**（沉积喷流型矿床）
- **VMS**（火山块状硫化物矿床）

编码规则：`pd.factorize` 按字母顺序 → 0 = SEDEX，1 = VMS

---

## 环境要求 / Requirements

- Python ≥ 3.8
- 详见 `requirements.txt`
