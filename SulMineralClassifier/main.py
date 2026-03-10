"""
main.py - SulMineralClassifier 交互式主程序
main.py - SulMineralClassifier Interactive Main Program

硫化矿物成因类型分类软件（SEDEX vs VMS）
Sulfide Mineral Genetic Type Classification Software (SEDEX vs VMS)

运行方式 / Usage:
    python main.py          (从 SulMineralClassifier/ 目录)
    python -m SulMineralClassifier.main   (从上级目录)
"""

import sys
import os

# 确保能以脚本方式直接运行 / Allow direct script execution
if __name__ == "__main__" and __package__ is None:
    # 将父目录加入路径，以支持相对导入
    # Add parent directory to path to support relative imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "SulMineralClassifier"

from .config import CLASS_NAMES, get_elements
from .core.data_loader import load_training_data, load_prediction_data
from .core.preprocessing import preprocess_data, encode_target, split_data
from .core.trainer import train_model, train_all_models
from .core.predictor import generate_report, save_report
from .visualization.confusion_plot import plot_confusion_matrix
from .explain.shap_analysis import run_shap_analysis


# ── 辅助函数 / Helper functions ───────────────────────────────────────────────

def _ask(prompt: str, choices: list) -> str:
    """
    循环提示用户输入，直到输入合法选项。
    Prompt user until a valid choice is entered.
    """
    choices_lower = [c.lower() for c in choices]
    while True:
        ans = input(prompt).strip().lower()
        if ans in choices_lower:
            return ans
        print(f"  ⚠  请输入以下选项之一 / Please enter one of: {choices}")


def _ask_path(prompt: str) -> str:
    """
    循环提示用户输入文件路径，直到路径合法。
    Prompt user for a file path until a valid path is entered.
    """
    while True:
        path = input(prompt).strip()
        if os.path.isfile(path):
            return path
        print(f"  ⚠  文件不存在，请重新输入 / File not found, please re-enter: {path}")


def _ask_dir(prompt: str) -> str:
    """
    提示用户输入目录路径（如目录不存在则自动创建）。
    Prompt user for a directory path (auto-created if not existing).
    """
    path = input(prompt).strip()
    if path:
        os.makedirs(path, exist_ok=True)
    return path or None


# ── 主程序 / Main program ─────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SulMineralClassifier — 硫化矿物成因类型分类软件")
    print("  SulMineralClassifier — Sulfide Mineral Genetic Type Classifier")
    print("  分类目标 / Classification target: SEDEX vs VMS")
    print("=" * 65)

    # ── 步骤 1：选择矿物类型 / Step 1: Select mineral type ───────────────────
    print(
        "\n步骤 1 / Step 1: 选择矿物类型 / Select mineral type"
        "\n  1 → 黄铜矿 Chalcopyrite (Ccp)"
        "\n  2 → 黄铁矿 Pyrite (Py)"
    )
    mineral_choice = _ask("请输入 1 或 2 / Enter 1 or 2: ", ["1", "2"])
    mineral_type = "ccp" if mineral_choice == "1" else "py"
    mineral_label = "黄铜矿 (Ccp)" if mineral_type == "ccp" else "黄铁矿 (Py)"
    elements = get_elements(mineral_type)
    print(f"\n  ✔ 已选择 / Selected: {mineral_label}")
    print(f"  使用特征列 / Feature columns: {elements}")

    # ── 步骤 2：加载训练数据 / Step 2: Load training data ────────────────────
    print(
        "\n步骤 2 / Step 2: 加载训练数据 / Load training data"
        "\n  请提供包含以下列的 Excel 文件："
        "\n  Please provide an Excel file containing these columns:"
        f"\n  'Deposit type', {', '.join(elements)}"
    )
    train_path = _ask_path("训练数据文件路径 / Training data file path: ")

    try:
        X, y_raw = load_training_data(train_path, mineral_type)
    except ValueError as e:
        print(f"\n  ✘ 数据加载失败 / Data loading failed:\n  {e}")
        sys.exit(1)

    # 预处理 / Preprocessing
    X, y_raw = preprocess_data(X, y_raw)
    y_encoded, categories = encode_target(y_raw)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # ── 步骤 3：选择模型 / Step 3: Select model ───────────────────────────────
    print(
        "\n步骤 3 / Step 3: 选择训练模型 / Select model to train"
        "\n  1 → MLP（多层感知机 / Multi-Layer Perceptron）"
        "\n  2 → RF（随机森林 / Random Forest）"
        "\n  3 → XGBoost"
        "\n  4 → Stacking（集成 / Ensemble）"
        "\n  5 → All（全部训练 / Train all）"
    )
    model_choice = _ask("请输入 1-5 / Enter 1-5: ", ["1", "2", "3", "4", "5"])
    model_map = {"1": "mlp", "2": "rf", "3": "xgb", "4": "stacking", "5": "all"}
    model_choice_name = model_map[model_choice]

    # ── 步骤 4：训练与评估 / Step 4: Train and evaluate ───────────────────────
    print("\n步骤 4 / Step 4: 训练模型并评估 / Training and evaluating model(s)...")
    if model_choice_name == "all":
        trained_models = train_all_models(X_train, X_test, y_train, y_test, mineral_type)
        # 让用户选择后续步骤（SHAP、预测）使用哪个模型
        # Let user pick which trained model to use for subsequent steps
        print(
            "\n所有模型训练完成。请选择用于后续预测和 SHAP 分析的模型：\n"
            "All models trained. Select the model for prediction and SHAP analysis:\n"
            "  1 → MLP\n  2 → RF\n  3 → XGBoost\n  4 → Stacking"
        )
        pick = _ask("请输入 1-4 / Enter 1-4: ", ["1", "2", "3", "4"])
        pick_map = {"1": "mlp", "2": "rf", "3": "xgb", "4": "stacking"}
        primary_model_name = pick_map[pick]
        primary_model = trained_models[primary_model_name]
        print(f"  ✔ 已选择 / Selected: {primary_model_name.upper()}")
    else:
        primary_model = train_model(
            model_choice_name, X_train, X_test, y_train, y_test, mineral_type
        )
        primary_model_name = model_choice_name
        trained_models = {model_choice_name: primary_model}

    # ── 可视化混淆矩阵 / Visualize confusion matrix ───────────────────────────
    print("\n正在绘制混淆矩阵 / Plotting confusion matrix...")
    save_viz = _ask(
        "是否保存混淆矩阵图片？/ Save confusion matrix figures? (y/n): ",
        ["y", "n"]
    )
    viz_dir = None
    if save_viz == "y":
        viz_dir = _ask_dir(
            "图片保存目录（留空则保存到当前目录）/ Save directory (leave blank for cwd): "
        )
        if not viz_dir:
            viz_dir = "."

    y_test_pred = primary_model.predict(X_test)
    plot_confusion_matrix(
        y_test, y_test_pred,
        class_names=CLASS_NAMES,
        model_name=primary_model_name.upper(),
        save_dir=viz_dir,
    )

    # ── 步骤 5：SHAP 分析 / Step 5: SHAP analysis ────────────────────────────
    print("\n步骤 5 / Step 5: SHAP 可解释性分析 / SHAP Explainability Analysis")
    do_shap = _ask(
        "是否进行 SHAP 分析？（计算较慢）/ Run SHAP analysis? (slow) (y/n): ",
        ["y", "n"]
    )
    if do_shap == "y":
        shap_dir = None
        save_shap = _ask("是否保存 SHAP 图片？/ Save SHAP figures? (y/n): ", ["y", "n"])
        if save_shap == "y":
            shap_dir = _ask_dir(
                "图片保存目录（留空则保存到当前目录）/ Save directory (leave blank for cwd): "
            )
            if not shap_dir:
                shap_dir = "."
        run_shap_analysis(
            primary_model,
            X_train, X_test,
            class_names=CLASS_NAMES,
            model_name=primary_model_name.upper(),
            save_dir=shap_dir,
        )

    # ── 步骤 6：预测新数据 / Step 6: Predict new data ────────────────────────
    print("\n步骤 6 / Step 6: 预测新数据 / Predict new data")
    do_predict = _ask(
        "是否加载新数据进行预测？/ Load new data for prediction? (y/n): ",
        ["y", "n"]
    )
    if do_predict == "y":
        pred_path = _ask_path("待预测数据文件路径 / Prediction data file path: ")
        try:
            X_pred = load_prediction_data(pred_path, mineral_type)
        except ValueError as e:
            print(f"\n  ✘ 数据加载失败 / Data loading failed:\n  {e}")
        else:
            if len(X_pred) == 0:
                print("  ⚠  没有可预测的有效样本 / No valid samples to predict.")
            else:
                report_df = generate_report(primary_model, X_pred)
                print("\n预测结果预览 / Prediction preview:")
                print(report_df.to_string(index=False))

                # ── 步骤 7：保存预测结果 / Step 7: Save prediction results ──────
                print("\n步骤 7 / Step 7: 保存预测结果 / Save prediction results")
                do_save = _ask(
                    "是否保存预测结果到 Excel？/ Save results to Excel? (y/n): ",
                    ["y", "n"]
                )
                if do_save == "y":
                    base = os.path.splitext(os.path.basename(pred_path))[0]
                    default_out = f"{base}_prediction_{primary_model_name}.xlsx"
                    out_path = input(
                        f"输出文件路径（回车使用默认：{default_out}）/ "
                        f"Output path (Enter for default: {default_out}): "
                    ).strip()
                    if not out_path:
                        out_path = default_out
                    save_report(report_df, out_path)

    print("\n" + "=" * 65)
    print("  程序运行完毕 / Program completed. Goodbye!")
    print("=" * 65)


if __name__ == "__main__":
    main()
