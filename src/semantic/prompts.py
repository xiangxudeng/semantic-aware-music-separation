"""标签提示词的统一构建。

第 2 周做长尾诊断时发现：对弱标签补同义与更具体的说法，比单纯换措辞有效得多。
例如 `eastern` 只写 "a sound of eastern" 时 AUC 只有 0.641；扩展成
"traditional Chinese music / Indian classical music / Asian traditional music" 之后升到 0.867。

所以把提示词构建收成一个模块：基础模板统一，弱标签额外补说法。
第 3 周的指令微调与条件分离也用这里的提示词，保证口径一致。
"""

from __future__ import annotations

BASE_TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
]

# 只给最弱的几类补同义/更具体的说法；其余标签只用基础模板。
# 注意 bass 扩展后反而略降（AUC 0.744 → 0.728），所以这里是逐标签可控的。
WEAK_TAG_EXTRA = {
    "instrumental": [
        "instrumental music",
        "music with no singing",
        "a track performed only by instruments",
    ],
    "modern": ["modern music", "contemporary music", "current-day popular music"],
    "eastern": [
        "traditional Chinese music",
        "Indian classical music",
        "Asian traditional music",
        "music with Eastern traditional instruments",
    ],
    "baroque": [
        "baroque music",
        "Baroque era classical music",
        "17th century classical music",
    ],
    "folk": ["folk music", "traditional folk songs", "acoustic folk music"],
    "upbeat": ["upbeat music", "cheerful and lively music", "happy fast tempo music"],
    "bass": ["deep bass", "low bass frequencies", "heavy bassline", "bass-heavy music"],
    "no vocals": ["music without vocals", "an instrumental track with no singing"],
}


def prompts_for_tag(tag: str, templates: list[str] | None = None, extras: dict | None = None) -> list[str]:
    """返回某个标签的全部提示词：基础模板 + 额外说法。"""
    templates = templates or BASE_TEMPLATES
    extras = WEAK_TAG_EXTRA if extras is None else extras
    return [t.format(tag) for t in templates] + list(extras.get(tag, []))
