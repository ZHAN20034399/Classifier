"""
generate_figures.py
===================
生成 SulMineralClassifier 三张说明图。
Generate the three documentation diagrams for SulMineralClassifier.

  图1  软件总体架构图         Software Overall Architecture Diagram
  图2  训练数据逻辑模型图     Training Data Logical Model Diagram
  图3  模型训练与预测数据流图  Model Training & Prediction Data Flow Diagram

Usage (run from any directory):
    python docs/figures/generate_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import fontManager

# ── CJK font setup ─────────────────────────────────────────────────────────────
_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK_PATH):
    fontManager.addfont(_CJK_PATH)
    CN = "Noto Sans CJK JP"
else:
    CN = "DejaVu Sans"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Low-level drawing helpers ─────────────────────────────────────────────────

def box(ax, cx, cy, w, h, fc, text, fs=9, bold=False, tc="white",
        radius=0.035, lw=0, ec="none", zorder=3):
    """Draw a rounded rectangle with centred multi-line text."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontfamily=CN, color=tc,
            fontweight="bold" if bold else "normal",
            zorder=zorder + 1, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color="#555", lw=1.5, hw=0.07, hl=0.07,
          zorder=2, cs="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"->,head_width={hw},head_length={hl}",
            color=color, lw=lw, connectionstyle=cs,
        ),
        zorder=zorder,
    )


def hline(ax, x0, x1, y, color="#aaa", lw=1.2):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=2)


def vline(ax, x, y0, y1, color="#aaa", lw=1.2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=2)


def band(ax, xlo, xhi, ylo, yhi, fc, ec="#CBD5E1", lw=1.0, zorder=0):
    """Light background band for a layer."""
    p = FancyBboxPatch(
        (xlo, ylo), xhi - xlo, yhi - ylo,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(p)


def diamond(ax, cx, cy, w, h, fc, text, fs=9, zorder=3):
    """Flowchart decision diamond."""
    pts = [
        [cx,         cy + h / 2],
        [cx + w / 2, cy],
        [cx,         cy - h / 2],
        [cx - w / 2, cy],
    ]
    poly = plt.Polygon(pts, closed=True, facecolor=fc,
                       edgecolor="none", zorder=zorder)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontfamily=CN, color="white",
            fontweight="bold", zorder=zorder + 1, linespacing=1.3)


# ══════════════════════════════════════════════════════════════════════════════
#  图1  软件总体架构图
# ══════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    W, H = 13, 9.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    # Colour palette
    CU = "#2563EB"   # User interface layer
    CC = "#0891B2"   # Config / control
    CP = "#0E7490"   # Core processing
    CM = "#059669"   # Models
    CO = "#D97706"   # Output
    BG = "#EFF6FF"   # Band background

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(W / 2, H - 0.40, "图1  SulMineralClassifier 软件总体架构图",
            ha="center", fontsize=15, fontfamily=CN,
            fontweight="bold", color="#1E293B")
    ax.text(W / 2, H - 0.80,
            "Software Overall Architecture Diagram",
            ha="center", fontsize=10, fontfamily=CN, color="#64748B")

    # ── Layer bands & labels ─────────────────────────────────────────────────
    # (yc, bh, cn_label, en_label, bg_color, border_color)
    layers = [
        (7.85, 0.82, "① 用户接口层", "User Interface Layer",    "#EFF6FF", "#BFDBFE"),
        (6.78, 0.82, "② 配置层",     "Configuration Layer",      "#F0FDF4", "#BBF7D0"),
        (5.42, 1.28, "③ 核心处理层", "Core Processing Layer",    "#F0F9FF", "#BAE6FD"),
        (3.72, 1.30, "④ 算法模型层", "Algorithm & Model Layer",  "#ECFDF5", "#A7F3D0"),
        (1.90, 1.18, "⑤ 输出与分析层","Output & Analysis Layer", "#FFFBEB", "#FDE68A"),
    ]
    for yc, bh, lcn, len_, bg, border in layers:
        band(ax, 0.28, W - 0.28, yc - bh / 2, yc + bh / 2, bg, border)
        ax.text(0.50, yc, f"{lcn}\n{len_}",
                ha="center", va="center", fontsize=7.2,
                fontfamily=CN, color="#475569", style="italic",
                linespacing=1.4)

    # ── Layer ①: User Interface ───────────────────────────────────────────
    box(ax, 3.5,  7.85, 2.7, 0.54, CU,
        "main.py\n交互式命令行界面 / Interactive CLI", fs=8.5, bold=True)
    box(ax, 9.5,  7.85, 2.7, 0.54, CU,
        "run.py  (选装)\n一键启动脚本 / Launcher script", fs=8.5)

    # ── Layer ②: Config ───────────────────────────────────────────────────
    box(ax, W / 2, 6.78, 7.2, 0.52, CC,
        "config.py  ·  超参数 | 特征列(Ccp 8列 / Py 10列) | 类别名称 | 随机种子\n"
        "Hyperparameters · Feature columns · Class names · Random seed",
        fs=8.5)

    # ── Layer ③: Core ─────────────────────────────────────────────────────
    core = [
        (1.80, "data_loader.py\n数据加载\nData Loader"),
        (4.30, "preprocessing.py\n数据预处理\nPreprocessing"),
        (6.95, "trainer.py\n模型训练\nTrainer"),
        (10.5, "predictor.py\n预测 & 导出\nPredictor"),
    ]
    for cx, lbl in core:
        box(ax, cx, 5.42, 2.30, 0.88, CP, lbl, fs=8.2)

    # ── Layer ④: Models ───────────────────────────────────────────────────
    models = [
        (2.10, "#2563EB", "mlp_model.py\nMLP 多层感知机"),
        (4.50, "#16A34A", "rf_model.py\nRF 随机森林"),
        (6.90, "#EA580C", "xgb_model.py\nXGBoost"),
        (9.30, "#7C3AED", "stacking_model.py\nStacking 集成"),
    ]
    for cx, fc, lbl in models:
        box(ax, cx, 3.72, 2.15, 0.80, fc, lbl, fs=8.2, bold=True)

    # ── Layer ⑤: Output ───────────────────────────────────────────────────
    outputs = [
        (2.50, CO,     "confusion_plot.py\n混淆矩阵可视化\nConfusion Matrix"),
        (6.50, "#0E7490","shap_analysis.py\nSHAP 可解释性分析\nSHAP Analysis"),
        (10.5, "#DC2626","Excel 预测报告\nPrediction Report (.xlsx)\n预测概率表"),
    ]
    for cx, fc, lbl in outputs:
        box(ax, cx, 1.90, 2.80, 0.90, fc, lbl, fs=8.2)

    # ── Arrows (vertical flow) ────────────────────────────────────────────
    # UI layer → Config
    arrow(ax, W / 2, 7.57, W / 2, 7.20, color=CC)

    # Config → Core (fan out)
    fan_y_top, fan_y_bot = 6.50, 6.00
    hline(ax, 1.80, 10.5, fan_y_top, color=CP)
    for cx, _ in core:
        vline(ax, cx, fan_y_top, fan_y_bot, color=CP)
        arrow(ax, cx, fan_y_bot, cx, 5.87, color=CP)

    # Core → Model (trainer drives models)
    tr_cx = 6.95
    merge_y = 4.38
    hline(ax, 2.10, 9.30, merge_y, color="#64748B", lw=1.0)
    for cx, fc, _ in models:
        vline(ax, cx, merge_y, 4.14, color=fc)
        arrow(ax, cx, 4.14, cx, 4.14, color=fc)

    # Model → Output
    out_merge = 2.57
    hline(ax, 2.50, 10.5, out_merge, color="#64748B", lw=1.0)
    for cx, fc, _ in outputs:
        vline(ax, cx, out_merge, 2.37, color=fc)
        arrow(ax, cx, 2.37, cx, 2.37, color=fc)

    # main.py ↔ run.py
    arrow(ax, 4.85, 7.85, 8.15, 7.85, color=CU)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        (CU,       "用户接口 User Interface"),
        (CC,       "配置 Config"),
        (CP,       "核心处理 Core"),
        ("#059669","算法模型 Models"),
        (CO,       "输出 Output"),
    ]
    patches = [mpatches.Patch(color=c, label=l) for c, l in legend_items]
    ax.legend(handles=patches, loc="lower right",
              prop={"family": CN, "size": 7.5},
              framealpha=0.9, edgecolor="#CBD5E1", ncol=3)

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "fig1_architecture.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[图1] 已保存 / Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  图2  训练数据逻辑模型图
# ══════════════════════════════════════════════════════════════════════════════

def fig2_data_model():
    W, H = 14, 10.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(W / 2, H - 0.42, "图2  训练数据逻辑模型图",
            ha="center", fontsize=15, fontfamily=CN,
            fontweight="bold", color="#1E293B")
    ax.text(W / 2, H - 0.85, "Training Data Logical Model Diagram",
            ha="center", fontsize=10, fontfamily=CN, color="#64748B")

    C_EXCEL  = "#6366F1"
    C_TARGET = "#DC2626"
    C_EL_CCP = "#0E7490"
    C_EL_PY  = "#92400E"
    C_SEDEX  = "#16A34A"
    C_VMS    = "#EA580C"
    C_SPLIT  = "#7C3AED"
    C_TRAIN  = "#1D4ED8"
    C_TEST   = "#9333EA"

    # ── Top: Excel source ─────────────────────────────────────────────────
    box(ax, W / 2, 9.12, 5.0, 0.62, C_EXCEL,
        "训练数据 Excel 文件 (Training Excel File)\n第一行为列名 / Header row defines column names",
        fs=9.5, bold=True)

    # fork to Ccp / Py branches
    ax.plot([W / 2, W / 2], [8.81, 8.50], color=C_EXCEL, lw=1.8, zorder=2)
    ax.plot([3.60, 10.40], [8.50, 8.50], color=C_EXCEL, lw=1.8, zorder=2)
    arrow(ax, 3.60,  8.50, 3.60,  8.10, color=C_EXCEL)
    arrow(ax, 10.40, 8.50, 10.40, 8.10, color=C_EXCEL)

    # ── Left panel: Ccp ───────────────────────────────────────────────────
    band(ax, 0.28, 6.72, 1.05, 8.15, "#EFF6FF", "#BFDBFE", lw=1.3)
    ax.text(3.60, 8.00, "黄铜矿 Chalcopyrite (Ccp)  ·  8 特征元素",
            ha="center", fontsize=10.5, fontfamily=CN,
            fontweight="bold", color="#1D4ED8")

    # Deposit type column
    box(ax, 3.60, 7.38, 6.0, 0.54, C_TARGET,
        "Deposit type  [必须列 Required]  →  'SEDEX'  |  'VMS'",
        fs=9, bold=True)

    # 8 feature element boxes  (2 rows × 4 cols)
    ccp_els = ["Co", "Ni", "Zn", "Cd", "Sb", "Pb", "Ag", "Se"]
    xs4 = [1.18, 2.58, 4.62, 6.02]
    ys2 = [6.74, 6.13]
    for i, el in enumerate(ccp_els):
        box(ax, xs4[i % 4], ys2[i // 4], 1.20, 0.46, C_EL_CCP,
            f"{el}  (ppm)", fs=8.8)
    ax.text(3.60, 5.78, "← 8 列数值型特征 →",
            ha="center", fontsize=8, fontfamily=CN,
            color="#0C4A6E", style="italic")

    # Encoding
    box(ax, 2.20, 5.25, 2.50, 0.50, C_SEDEX, "SEDEX → 0", fs=9.5, bold=True)
    box(ax, 5.00, 5.25, 2.50, 0.50, C_VMS,   "VMS   → 1", fs=9.5, bold=True)
    ax.text(3.60, 4.88,
            "pd.factorize(sort=True)  按字母顺序编码 / alphabetical encoding",
            ha="center", fontsize=8, fontfamily=CN, color="#475569")

    # Split
    box(ax, 3.60, 4.44, 6.0, 0.54, C_SPLIT,
        "数据集划分 Train/Test Split  (test_size=0.2, random_state=2)",
        fs=9)
    box(ax, 2.00, 3.72, 2.60, 0.54, C_TRAIN,
        "训练集 Train Set\n约 80% 样本 (~80 rows)", fs=8.5, bold=True)
    box(ax, 5.20, 3.72, 2.60, 0.54, C_TEST,
        "测试集 Test Set\n约 20% 样本 (~20 rows)", fs=8.5, bold=True)

    box(ax, 2.00, 2.94, 2.60, 0.50, "#DBEAFE",
        "model.fit(X_train, y_train)", fs=8, tc="#1E3A8A",
        ec="#93C5FD", lw=1)
    box(ax, 5.20, 2.94, 2.60, 0.50, "#EDE9FE",
        "classification_report\nconfusion_matrix", fs=8, tc="#4C1D95",
        ec="#C4B5FD", lw=1)

    # Ccp inner arrows
    arrow(ax, 3.60, 7.11, 3.60, 6.99, color=C_TARGET)
    arrow(ax, 3.60, 5.91, 3.60, 5.52, color="#0C4A6E")
    arrow(ax, 3.60, 5.00, 3.60, 4.73, color=C_SPLIT)
    arrow(ax, 2.80, 4.18, 2.20, 4.00, color=C_TRAIN)
    arrow(ax, 4.40, 4.18, 5.20, 4.00, color=C_TEST)
    arrow(ax, 2.00, 3.45, 2.00, 3.21, color=C_TRAIN)
    arrow(ax, 5.20, 3.45, 5.20, 3.21, color=C_TEST)

    # ── Right panel: Py ───────────────────────────────────────────────────
    band(ax, 7.28, 13.72, 1.05, 8.15, "#FFF7ED", "#FED7AA", lw=1.3)
    ax.text(10.40, 8.00, "黄铁矿 Pyrite (Py)  ·  10 特征元素",
            ha="center", fontsize=10.5, fontfamily=CN,
            fontweight="bold", color="#C2410C")

    box(ax, 10.40, 7.38, 6.0, 0.54, C_TARGET,
        "Deposit type  [必须列 Required]  →  'SEDEX'  |  'VMS'",
        fs=9, bold=True)

    # 10 feature element boxes  (2 rows × 5 cols)
    py_els = ["Co", "Ni", "Zn", "Cu", "Sb", "Pb", "Ag", "Se", "As", "Bi"]
    xs5 = [8.12, 9.22, 10.40, 11.58, 12.68]
    for i, el in enumerate(py_els):
        box(ax, xs5[i % 5], ys2[i // 5], 1.0, 0.46, C_EL_PY,
            f"{el}  (ppm)", fs=8.8)
    ax.text(10.40, 5.78, "← 10 列数值型特征 →",
            ha="center", fontsize=8, fontfamily=CN,
            color="#7C2D12", style="italic")

    box(ax, 9.00, 5.25, 2.50, 0.50, C_SEDEX, "SEDEX → 0", fs=9.5, bold=True)
    box(ax, 11.80, 5.25, 2.50, 0.50, C_VMS,   "VMS   → 1", fs=9.5, bold=True)
    ax.text(10.40, 4.88,
            "pd.factorize(sort=True)  按字母顺序编码 / alphabetical encoding",
            ha="center", fontsize=8, fontfamily=CN, color="#475569")

    box(ax, 10.40, 4.44, 6.0, 0.54, C_SPLIT,
        "数据集划分 Train/Test Split  (test_size=0.2, random_state=2)",
        fs=9)
    box(ax, 8.80, 3.72, 2.60, 0.54, C_TRAIN,
        "训练集 Train Set\n约 80% 样本 (~80 rows)", fs=8.5, bold=True)
    box(ax, 12.00, 3.72, 2.60, 0.54, C_TEST,
        "测试集 Test Set\n约 20% 样本 (~20 rows)", fs=8.5, bold=True)
    box(ax, 8.80, 2.94, 2.60, 0.50, "#DBEAFE",
        "model.fit(X_train, y_train)", fs=8, tc="#1E3A8A",
        ec="#93C5FD", lw=1)
    box(ax, 12.00, 2.94, 2.60, 0.50, "#EDE9FE",
        "classification_report\nconfusion_matrix", fs=8, tc="#4C1D95",
        ec="#C4B5FD", lw=1)

    arrow(ax, 10.40, 7.11, 10.40, 6.99, color=C_TARGET)
    arrow(ax, 10.40, 5.91, 10.40, 5.52, color="#7C2D12")
    arrow(ax, 10.40, 5.00, 10.40, 4.73, color=C_SPLIT)
    arrow(ax, 9.60,  4.18, 9.00,  4.00, color=C_TRAIN)
    arrow(ax, 11.20, 4.18, 12.00, 4.00, color=C_TEST)
    arrow(ax, 8.80, 3.45, 8.80, 3.21, color=C_TRAIN)
    arrow(ax, 12.00, 3.45, 12.00, 3.21, color=C_TEST)

    # ── Bottom note ───────────────────────────────────────────────────────
    ax.text(W / 2, 1.52,
            "★  所有特征列为数值型 (ppm)  ·  系统自动删除含缺失值的行",
            ha="center", fontsize=9, fontfamily=CN, color="#7C3AED")
    ax.text(W / 2, 1.17,
            "All feature columns are numeric (ppm)  ·  "
            "Rows with missing values are dropped automatically",
            ha="center", fontsize=8.5, fontfamily=CN, color="#64748B")

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "fig2_data_model.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[图2] 已保存 / Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  图3  模型训练与预测数据流图
# ══════════════════════════════════════════════════════════════════════════════

def fig3_dataflow():
    W, H = 11, 18
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    cx = W / 2

    # Colour palette
    C_IO   = "#6366F1"
    C_PROC = "#0891B2"
    C_DEC  = "#F59E0B"
    C_FIT  = "#059669"
    C_EVAL = "#7C3AED"
    C_VIZ  = "#DC2626"
    C_SHAP = "#0E7490"
    C_OUT  = "#D97706"
    C_ARR  = "#64748B"

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(cx, H - 0.45, "图3  模型训练与预测数据流图",
            ha="center", fontsize=15, fontfamily=CN,
            fontweight="bold", color="#1E293B")
    ax.text(cx, H - 0.90,
            "Model Training and Prediction Data Flow Diagram",
            ha="center", fontsize=10, fontfamily=CN, color="#64748B")

    # ── Phase labels (left gutter) ────────────────────────────────────────
    def phase_label(ax, y, label, sub, fc, ec):
        ax.text(0.52, y, f"{label}\n{sub}",
                ha="center", va="center", fontsize=7.5,
                fontfamily=CN, color=fc, fontweight="bold",
                linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.25", fc=fc + "22",
                          ec=ec, lw=1.1))

    # ── Helper: draw a centered flowchart node with arrow from above ──────
    prev = [None]   # mutable state for chaining

    def node(y, w, h, fc, text, fs=9, bold=False, no_in_arrow=False):
        if prev[0] is not None and not no_in_arrow:
            arrow(ax, cx, prev[0] - 0.02, cx, y + h / 2 + 0.02, color=C_ARR)
        box(ax, cx, y, w, h, fc, text, fs=fs, bold=bold)
        prev[0] = y - h / 2

    # ════ TRAINING PHASE ═════════════════════════════════════════════════
    # 1. Start
    node(16.90, 3.5, 0.55, "#1E293B", "开始 / Start", fs=10, bold=True,
         no_in_arrow=True)

    # 2. Load training data
    node(16.05, 7.0, 0.65, C_IO,
         "输入：训练数据 Excel 文件\nInput: Training Excel File\n"
         "(包含 Deposit type + 特征元素列)")

    # 3. Select mineral type
    node(15.08, 7.0, 0.65, C_IO,
         "选择矿物类型 / Select Mineral Type\n"
         "Ccp (黄铜矿, 8 特征)  |  Py (黄铁矿, 10 特征)")

    # 4. Load & validate columns
    node(14.10, 7.0, 0.68, C_PROC,
         "data_loader: 读取 Excel → 校验并提取特征列 + 标签列\n"
         "Read Excel → Validate & extract feature & label columns")

    # 5. Preprocess
    node(13.10, 7.0, 0.68, C_PROC,
         "preprocessing: 删除缺失行 · 数值强制转换\n"
         "Drop NaN rows · Coerce all columns to numeric")

    # 6. Encode
    node(12.10, 7.0, 0.68, C_PROC,
         "encode_target: pd.factorize(sort=True)\n"
         "SEDEX → 0  ·  VMS → 1")

    # 7. Split
    node(11.10, 7.0, 0.68, C_PROC,
         "split_data: 训练集 80%  /  测试集 20%\n"
         "Train 80% / Test 20%  (random_state=2)")

    # 8. Select classifier
    node(10.08, 7.0, 0.68, C_IO,
         "选择分类器 / Select Classifier\n"
         "MLP  |  RF  |  XGBoost  |  Stacking  |  All")

    # 9. Four parallel model boxes
    model_y = 8.90
    ml_w, ml_h = 2.05, 0.80
    ml_data = [
        (1.80, "#2563EB", "MLP\n多层感知机"),
        (4.05, "#16A34A", "RF\n随机森林"),
        (6.95, "#EA580C", "XGBoost"),
        (9.20, "#7C3AED", "Stacking\n集成"),
    ]
    # fan-out arrow from node above
    fan_top = prev[0] - 0.02
    fan_sep = model_y + ml_h / 2 + 0.12
    ax.plot([cx, cx], [fan_top, fan_sep + 0.18], color=C_ARR, lw=1.5, zorder=2)
    hline(ax, ml_data[0][0], ml_data[-1][0], fan_sep + 0.18, C_ARR)
    for mx, mc, lbl in ml_data:
        vline(ax, mx, fan_sep + 0.18, model_y + ml_h / 2, mc)
        box(ax, mx, model_y, ml_w, ml_h, mc, lbl, fs=9, bold=True)

    # fan-in after models
    merge_y = model_y - ml_h / 2 - 0.12
    hline(ax, ml_data[0][0], ml_data[-1][0], merge_y, C_ARR)
    for mx, mc, _ in ml_data:
        vline(ax, mx, model_y - ml_h / 2, merge_y, mc)
    arrow(ax, cx, merge_y, cx, merge_y, color=C_ARR)
    prev[0] = merge_y

    # 10. Fit
    node(7.68, 7.0, 0.66, C_FIT,
         "model.fit(X_train, y_train)  模型拟合 / Model Fitting")

    # 11. Evaluate
    node(6.72, 7.0, 0.70, C_EVAL,
         "评估 / Evaluate: classification_report + confusion_matrix\n"
         "训练集报告 + 测试集报告 / Train & Test reports")

    # 12. Confusion matrix
    node(5.72, 7.0, 0.68, C_VIZ,
         "可视化混淆矩阵 / Plot Confusion Matrix\n"
         "(可保存为 PNG / SVG / PDF)")

    # ════ ANALYSIS PHASE — SHAP decision ════════════════════════════════
    shap_dec_y = 4.90
    arrow(ax, cx, prev[0] - 0.02, cx, shap_dec_y + 0.34, color=C_ARR)
    diamond(ax, cx, shap_dec_y, 4.8, 0.68, C_DEC,
            "是否进行 SHAP 分析?\nRun SHAP Analysis?", fs=9)
    prev[0] = shap_dec_y - 0.34

    # Yes branch
    shap_box_y = 4.06
    arrow(ax, cx, shap_dec_y - 0.34, cx, shap_box_y + 0.34, color=C_SHAP)
    ax.text(cx + 0.22, (shap_dec_y + shap_box_y) / 2,
            "是/Yes", fontsize=8, fontfamily=CN, color=C_SHAP)
    box(ax, cx, shap_box_y, 7.0, 0.66, C_SHAP,
        "shap_analysis: KernelExplainer → Beeswarm 特征重要性图\n"
        "Feature importance beeswarm plot per class")
    prev[0] = shap_box_y - 0.33

    # No branch (right bypass to pred-decision)
    no_x = 9.80
    pred_dec_y = 3.02  # defined below
    ax.annotate("", xy=(no_x, shap_dec_y),
                xytext=(cx + 2.4, shap_dec_y),
                arrowprops=dict(arrowstyle="-", color=C_DEC, lw=1.5))
    ax.text(8.0, shap_dec_y + 0.15, "否/No", fontsize=8,
            fontfamily=CN, color=C_DEC, ha="center")

    # ════ PREDICTION PHASE ════════════════════════════════════════════════
    arrow(ax, cx, prev[0] - 0.02, cx, pred_dec_y + 0.34, color=C_ARR)
    # reconnect No bypass
    ax.annotate("", xy=(no_x, pred_dec_y),
                xytext=(no_x, shap_dec_y),
                arrowprops=dict(arrowstyle="-", color=C_DEC, lw=1.5))
    ax.annotate("", xy=(cx + 2.4, pred_dec_y),
                xytext=(no_x, pred_dec_y),
                arrowprops=dict(arrowstyle="->", color=C_DEC, lw=1.5))

    diamond(ax, cx, pred_dec_y, 4.8, 0.68, C_DEC,
            "是否预测新数据?\nPredict New Data?", fs=9)
    prev[0] = pred_dec_y - 0.34

    # Yes branch — load prediction data
    pred1_y = 2.15
    arrow(ax, cx, pred_dec_y - 0.34, cx, pred1_y + 0.32, color=C_PROC)
    ax.text(cx + 0.20, (pred_dec_y + pred1_y) / 2,
            "是/Yes", fontsize=8, fontfamily=CN, color=C_PROC)
    box(ax, cx, pred1_y, 7.0, 0.62, C_PROC,
        "data_loader: 加载待预测 Excel → 校验特征列 → 删除缺失行\n"
        "Load prediction Excel → validate columns → drop NaN rows")
    prev[0] = pred1_y - 0.31

    # Predict + export
    pred2_y = 1.32
    arrow(ax, cx, prev[0] - 0.01, cx, pred2_y + 0.32, color=C_OUT)
    box(ax, cx, pred2_y, 7.0, 0.62, C_OUT,
        "predict + predict_proba → generate_report → save Excel (.xlsx)\n"
        "预测类别 + 概率 → 报告 → 保存结果文件", bold=True)
    prev[0] = pred2_y - 0.31

    # End
    end_y = 0.44
    arrow(ax, cx, prev[0] - 0.01, cx, end_y + 0.24, color=C_ARR)
    box(ax, cx, end_y, 3.2, 0.46, "#1E293B",
        "结束 / End", fs=10, bold=True)

    # No branch bypass → End
    no_end_x = 0.60
    ax.annotate("", xy=(no_end_x, pred_dec_y),
                xytext=(cx - 2.4, pred_dec_y),
                arrowprops=dict(arrowstyle="-", color=C_DEC, lw=1.5))
    ax.annotate("", xy=(no_end_x, end_y),
                xytext=(no_end_x, pred_dec_y),
                arrowprops=dict(arrowstyle="-", color=C_DEC, lw=1.5))
    ax.annotate("", xy=(cx - 1.6, end_y),
                xytext=(no_end_x, end_y),
                arrowprops=dict(arrowstyle="->", color=C_DEC, lw=1.5))
    ax.text(0.18, (pred_dec_y + end_y) / 2, "否\nNo",
            fontsize=8, fontfamily=CN, color=C_DEC, ha="center")

    # ── Phase background bands ─────────────────────────────────────────────
    # Training
    band(ax, 0.28, W - 0.28, 7.22, 17.30, "#EFF6FF", "#BFDBFE", lw=0, zorder=0)
    ax.text(0.48, 12.30, "训练阶段\nTraining\nPhase",
            ha="center", fontsize=7.5, fontfamily=CN, color="#1D4ED8",
            fontweight="bold")
    # Analysis
    band(ax, 0.28, W - 0.28, 3.62, 7.18, "#F0FDF4", "#BBF7D0", lw=0, zorder=0)
    ax.text(0.48, 5.40, "分析阶段\nAnalysis\nPhase",
            ha="center", fontsize=7.5, fontfamily=CN, color="#15803D",
            fontweight="bold")
    # Prediction
    band(ax, 0.28, W - 0.28, 0.18, 3.58, "#FFF7ED", "#FED7AA", lw=0, zorder=0)
    ax.text(0.48, 1.88, "预测阶段\nPrediction\nPhase",
            ha="center", fontsize=7.5, fontfamily=CN, color="#C2410C",
            fontweight="bold")

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "fig3_dataflow.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[图3] 已保存 / Saved: {out}")
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("正在生成三张说明图 / Generating three diagrams ...")
    fig1_architecture()
    fig2_data_model()
    fig3_dataflow()
    print("全部完成 / All done.")
