from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSLATION_PATH = ROOT / "translation.json"
SUPPORTED_OUTPUT_LANGUAGES = ("chinese", "english")

def normalize_output_language(value: str) -> str:
    normalized = str(value).strip().lower()
    alias_map = {
        "zh": "chinese",
        "cn": "chinese",
        "chinese": "chinese",
        "中文": "chinese",
        "en": "english",
        "eng": "english",
        "english": "english",
        "英文": "english",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported output language: {value}")
    return alias_map[normalized]

def resolve_default_output_root(base_path: str, output_language: str) -> str:
    normalized_language = normalize_output_language(output_language)
    if normalized_language != "english":
        return base_path

    path_obj = Path(base_path)
    if path_obj.name:
        english_name = f"{path_obj.name}_english"
        return str(path_obj.with_name(english_name))
    return f"{base_path}_english"

def _contains_non_ascii(value: str) -> bool:
    return any(ord(char) > 127 for char in value)

def _looks_like_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith("/mnt/") or re.match(r"^[A-Za-z]:\\", text):
        return True
    if "://" in text:
        return True
    if "\\" in text:
        return True
    if text.startswith("/") and re.search(r"/[^/\s]+\.[A-Za-z0-9]+$", text):
        return True
    if "/" in text and re.match(r"^(?:\./|\.\./)?(?:[^/\s]+/)+[^/\s]+\.[A-Za-z0-9]+$", text):
        return True
    return False

def _apply_dynamic_phrase_rewrites(value: str) -> str:
    rewritten = value
    rewritten = re.sub(
        r"分组特征'([^']+)'有(\d+)行缺失值未纳入任何分组；如需单独成组，请在分组JSON中设置 include_null=true",
        r"Grouping feature '\1' has \2 rows with missing values excluded from all groups; set include_null=true in the grouping JSON to create a dedicated null group",
        rewritten,
    )
    rewritten = re.sub(
        r"Grouping feature '([^']+)'有(\d+) rows with missing values were excluded from all groups; set include_null=true in the grouping JSON to create a dedicated null group",
        r"Grouping feature '\1' has \2 rows with missing values excluded from all groups; set include_null=true in the grouping JSON to create a dedicated null group",
        rewritten,
    )
    def _replace_rare_group_warning(match: re.Match[str]) -> str:
        merged_group = "Other" if match.group(3) == "其他" else match.group(3)
        return (
            f"Joint grouping contains {match.group(1)} rare groups with prevalence below {match.group(2)}, "
            f"merged into '{merged_group}' group (total {match.group(4)} rows, {match.group(5)}%)"
        )

    rewritten = re.sub(
        r"联合分组中有(\d+)个稀有组占比低于 ([^，]+)，已合并为'([^']+)'组（合计 ([0-9/]+) 行，([0-9.]+)%）",
        _replace_rare_group_warning,
        rewritten,
    )
    return rewritten

class OutputLocalizer:
    def __init__(
        self,
        output_language: str = "chinese",
        translation_path: str | Path | None = None,
    ) -> None:
        self.output_language = normalize_output_language(output_language)
        self.translation_path = Path(translation_path) if translation_path is not None else DEFAULT_TRANSLATION_PATH
        self.exact_map: dict[str, str] = {}
        self.substring_keys: list[str] = []

        if self.output_language == "english":
            with self.translation_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for section_name in ("columns", "terms", "phrases"):
                section = payload.get(section_name, {})
                if isinstance(section, dict):
                    self.exact_map.update({str(key): str(value) for key, value in section.items()})
            self.substring_keys = sorted(
                [key for key in self.exact_map if _contains_non_ascii(key)],
                key=len,
                reverse=True,
            )

    @property
    def is_english(self) -> bool:
        return self.output_language == "english"

    def sanitize_filename(self, value: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*]+', "_", str(value))
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("._")
        return sanitized or "unnamed"

    def translate_text(self, value: str) -> str:
        if not self.is_english:
            return value
        if value in self.exact_map:
            return self.exact_map[value]
        translated = value
        for key in self.substring_keys:
            translated = translated.replace(key, self.exact_map[key])
        return _apply_dynamic_phrase_rewrites(translated)

    def translate_cell(self, value: Any) -> Any:
        if not self.is_english or not isinstance(value, str):
            return value
        if _looks_like_path(value):
            return value
        return self.translate_text(value)

    def translate_json(self, value: Any) -> Any:
        if not self.is_english:
            return value
        if isinstance(value, dict):
            translated: dict[str, Any] = {}
            for key, sub_value in value.items():
                new_key = self.translate_text(key) if isinstance(key, str) and not _looks_like_path(key) else key
                translated[str(new_key)] = self.translate_json(sub_value)
            return translated
        if isinstance(value, list):
            return [self.translate_json(item) for item in value]
        if isinstance(value, str):
            return self.translate_cell(value)
        return value

    def localize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_english or df.empty:
            return df.copy()
        localized = df.copy()
        localized.columns = [self.translate_text(str(column)) for column in localized.columns]
        for column in localized.columns:
            if pd.api.types.is_object_dtype(localized[column]) or pd.api.types.is_string_dtype(localized[column]):
                localized[column] = localized[column].map(self.translate_cell)
        return localized

    def localize_matrix_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_english or df.empty:
            return df.copy()
        localized = df.copy()
        localized.columns = [self.translate_text(str(column)) for column in localized.columns]
        localized.index = [self.translate_text(str(index)) for index in localized.index]
        return localized

    def save_csv(self, df: pd.DataFrame, path: str, *, index: bool = False, matrix: bool = False) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        localized_df = self.localize_matrix_dataframe(df) if matrix else self.localize_dataframe(df)
        localized_df.to_csv(path, index=index, encoding="utf-8-sig")

    def save_json(self, payload: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        localized_payload = self.translate_json(payload)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(localized_payload, handle, ensure_ascii=False, indent=2)

    def configure_matplotlib(self) -> None:
        try:
            import matplotlib
            from matplotlib import font_manager
            from cycler import cycler
        except ImportError:
            return

        common_rc = {
            "axes.unicode_minus": False,
            "axes.edgecolor": "#4a4a4a",
            "axes.linewidth": 0.7,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "regular",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#3d3d3d",
            "ytick.color": "#3d3d3d",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.25,
            "patch.linewidth": 0.4,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "axes.prop_cycle": cycler(color=["#2F4B7C", "#B07D4F", "#4F6D7A", "#8B3A3A", "#6C757D"]),
        }
        matplotlib.rcParams.update(common_rc)

        if self.is_english:
            preferred_fonts = [
                "Times New Roman",
                "Times New Roman PS MT",
                "Nimbus Roman",
                "TeX Gyre Termes",
                "DejaVu Serif",
            ]
            available = {font.name for font in font_manager.fontManager.ttflist}
            resolved = [font for font in preferred_fonts if font in available] or preferred_fonts
            matplotlib.rcParams["font.family"] = ["serif"]
            matplotlib.rcParams["font.serif"] = resolved
            return

        preferred_fonts = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        available_font_names = {font.name for font in font_manager.fontManager.ttflist}
        resolved_fonts = [font_name for font_name in preferred_fonts if font_name in available_font_names]
        if not resolved_fonts:
            resolved_fonts = preferred_fonts
        matplotlib.rcParams["font.family"] = ["sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = resolved_fonts
