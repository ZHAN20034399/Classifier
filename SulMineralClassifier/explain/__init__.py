"""
explain/__init__.py

导出可解释性分析模块的公共接口。
Exports the public interface of the explain module.
"""

from .shap_analysis import run_shap_analysis

__all__ = ["run_shap_analysis"]
