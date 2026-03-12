"""
visualization/__init__.py

导出可视化模块的公共接口。
Exports the public interface of the visualization module.
"""

from .confusion_plot import plot_confusion_matrix

__all__ = ["plot_confusion_matrix"]
