"""音源分离相关模块：条件注入（FiLM）、微调、评测。"""

from .film import FiLM, ConditionedModel, attach_film, infer_channel_dims

__all__ = ["FiLM", "ConditionedModel", "attach_film", "infer_channel_dims"]
