"""可运行的四轨分离、语义理解、文字检索与创作建议。"""

from pathlib import Path
import json
import os
import threading
import time
import uuid
import numpy as np
import pandas as pd

TEMPLATES = [
    "This is a sound of {}",
    "This audio contains {}",
    "This is a music track with {}",
    "A music piece featuring {}",
]


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def topk(embeddings, query, k=5):
    """余弦排序采用稳定次序；零向量不能作为有效检索请求。"""
    q = normalize(query).reshape(-1)
    if not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("无有效查询向量")
    scores = normalize(embeddings) @ q
    idx = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
    return idx, scores[idx]


def format_creation(analysis, raw):
    """固定四轨覆盖；模型草稿不合格时明确使用规则草稿，不伪装生成成功。"""
    tracks = analysis.get("tracks", {})
    keys = ["vocals", "drums", "bass", "other"]
    if any(key not in tracks for key in keys):
        raise ValueError("分析结果必须包含人声、鼓、贝斯、其他四轨")
    labels = [t["tag"] for t in analysis.get("whole_tags", [])]
    # ponytail: 两类候选标签规则只作保底；有独立试听证据后再扩充场景规则。
    if set(labels) & {"rock", "metal", "hard rock", "heavy", "fast", "dance", "techno"}:
        caption, scenes = (
            "让镜头跟着音乐向前，把今天的热爱留在画面里。",
            ["运动剪辑", "城市旅行", "活动转场"],
        )
    else:
        caption, scenes = (
            "把日常片段串成故事，让音乐陪这一刻慢慢展开。",
            ["生活记录", "旅行风景", "日常回忆"],
        )
    source, reason = "Qwen结构化草稿", ""
    try:
        draft, _ = json.JSONDecoder().raw_decode(raw[raw.index("{") :])
        new_caption, new_scenes = draft["caption"], draft["scenes"]
        if not isinstance(new_caption, str) or not 6 <= len(new_caption) <= 100:
            raise ValueError("文案长度或类型不合格")
        new_caption = new_caption.strip()
        if (
            not isinstance(new_scenes, list)
            or len(new_scenes) != 3
            or any(not isinstance(s, str) or not 2 <= len(s) <= 16 for s in new_scenes)
        ):
            raise ValueError("需要三个简短场景")
        new_scenes = [s.strip() for s in new_scenes]
        if len(new_caption) < 6 or any(len(s) < 2 for s in new_scenes):
            raise ValueError("文案或场景为空白")
        if len(set(new_scenes)) != 3:
            raise ValueError("场景重复")
        combined = new_caption + " ".join(new_scenes)
        if any(
            word.lower() in combined.lower()
            for word in [
                "混音",
                "均衡",
                "教学",
                "教程",
                "课程",
                "观众",
                "制作人",
                "学习资源",
                "大师",
                "EQ",
                "RMS",
                "BPM",
                "dB",
                "赫兹",
                "已处理",
                "已经",
                "已完成",
                "乐器",
                "人声",
                "鼓点",
                "贝斯",
                "吉他",
                "钢琴",
            ]
        ):
            raise ValueError("草稿含技术讲解、处理声明或未经确认的声音事实")
        caption, scenes = new_caption.strip(), [s.strip() for s in new_scenes]
    except (ValueError, KeyError, TypeError) as exc:
        source, reason = "候选标签规则草稿", str(exc)
    names = ["人声", "鼓", "贝斯", "其他"]
    advice = [
        "若希望主唱更靠前，先试听并调整它与伴奏的相对音量。",
        "若鼓遮盖主唱，先比较两轨音量；需要节奏突出时反向调整。",
        "若低频与鼓重叠，先试听两轨叠加，再决定音量或均衡调整。",
        "若伴奏遮盖主唱，可比较降低音量或调整声像后的效果。",
    ]
    lines = []
    for key, name, tip in zip(keys, names, advice):
        tags = tracks[key].get("tags", [])[:3]
        if tracks[key].get("rms", 0) <= 1e-7 or not tags:
            lines.append(
                f"- {name}：轨道近乎静音或无候选标签，先试听确认，不自动增益补偿。"
            )
        else:
            candidates = "、".join(t["tag"] for t in tags)
            lines.append(f"- {name}（候选：{candidates}）：{tip}")
    text = "创作草稿：候选标签需试听核对，尚未应用混音处理。\n\n混音建议\n" + "\n".join(
        lines
    )
    text += (
        "\n\n短视频文案草稿\n"
        + caption
        + "\n\n场景推荐（可尝试）\n"
        + "\n".join("- " + s for s in scenes)
    )
    text += "\n\n文案与场景来源：" + source
    if reason:
        text += "（模型输出未通过格式或内容校验）"
    return dict(text=text, source=source, reason=reason, caption=caption, scenes=scenes)


class MusicSystem:
    def __init__(self, assets=None, device="auto", output=None):
        import torch

        self.assets = Path(
            assets
            or os.environ.get(
                "AI_MUSIC_ASSETS", Path(__file__).resolve().parent.parent / "ai_music"
            )
        ).resolve()
        self.output = Path(
            output or Path(__file__).resolve().parent / "outputs"
        ).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.encoder = None
        self.separator = None
        self.llm = None
        self.tokenizer = None
        self.lock = (
            threading.RLock()
        )  # ponytail: 单锁串行推理；需要多人吞吐时改为独立模型进程。
        self.tags = pd.read_csv(self.assets / "results" / "mtt_50tags.csv").tag.tolist()
        self.tag_vectors = None
        self.index = pd.read_csv(
            self.assets / "results" / "mtt_index.csv", dtype={"clip_id": str}
        )
        self.embeddings = np.load(
            self.assets / "results" / "clap_official_emb_all25863_w3_fp16.npy",
            mmap_mode="r",
        )
        if len(self.index) != len(self.embeddings):
            raise ValueError("索引和音频向量数量不同")
        # 缓存必须沿用原始 annotations 顺序；不能对索引排序后直接复用向量。
        ann = pd.read_csv(
            self.assets / "data" / "mtt" / "annotations_final.csv",
            sep="\t",
            usecols=["clip_id"],
            dtype=str,
        )
        if not np.array_equal(ann.clip_id.to_numpy(), self.index.clip_id.to_numpy()):
            raise ValueError("缓存向量与索引 clip_id 未对齐")

    def _encoder(self):
        if self.encoder is None:
            # 复用已验证的官方 CLAP 模块，精简交付中只保留调用到的编码接口。
            from encoder import ClapEncoder

            self.encoder = ClapEncoder(self.assets, self.device)
        return self.encoder

    def tag_audio(self, path):
        with self.lock:
            e = self._encoder()
            if self.tag_vectors is None:
                self.tag_vectors = normalize(
                    np.mean(
                        [e.text([t.format(s) for s in self.tags]) for t in TEMPLATES],
                        axis=0,
                    )
                )
            vector = e.audio(path)
            idx, score = topk(self.tag_vectors, vector)
            return [
                dict(tag=self.tags[i], similarity=float(s)) for i, s in zip(idx, score)
            ]

    def search(self, text, k=5):
        if text is None or not str(text).strip():
            raise ValueError("请输入音乐描述")
        with self.lock:
            q = self._encoder().text([str(text).strip()])[0]
        idx, score = topk(self.embeddings, q, k)
        rows = []
        for i, s in zip(idx, score):
            row = self.index.iloc[i]
            rows.append(
                dict(
                    clip_id=str(row.clip_id),
                    tags="未标注" if pd.isna(row.tags) else str(row.tags),
                    similarity=float(s),
                    path=str(self.assets / "data" / "mtt" / row.mp3_path),
                )
            )
        return rows

    def analyze(self, path):
        import soundfile as sf
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        path = Path(path) if path else None
        if path is None or not path.is_file():
            raise ValueError("请提供可读音频文件")
        info = sf.info(path)
        if info.duration <= 0 or info.duration > 600:
            raise ValueError("音频长度应在 0 到 600 秒之间")
        if info.channels not in [1, 2]:
            raise ValueError("仅支持单声道或双声道音频")
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-7:
            raise ValueError("音频含无效数值或为静音")
        start = time.perf_counter()
        with self.lock:
            if self.separator is None:
                self.separator = get_model("htdemucs").to(self.device).eval()
            model = self.separator
            wav = torch.from_numpy(audio.T.copy())
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            mono = wav.mean(0)
            mean = mono.mean()
            std = mono.std().clamp_min(1e-8)
            with torch.inference_mode():
                separated = (
                    apply_model(
                        model,
                        ((wav - mean) / std)[None],
                        device=self.device,
                        shifts=0,
                        split=True,
                        overlap=0.25,
                    )[0].cpu()
                    * std
                    + mean
                )
            folder = self.output / uuid.uuid4().hex
            folder.mkdir()
            tracks = {}
            for stem, w in zip(model.sources, separated):
                target = folder / f"{stem}.wav"
                sf.write(target, w.T.numpy(), model.samplerate, subtype="FLOAT")
                rms = float(torch.sqrt(torch.mean(w * w)))
                tags = self.tag_audio(target) if rms > 1e-7 else []
                tracks[stem] = dict(
                    path=str(target),
                    rms=rms,
                    tags=tags,
                    description="候选语义：" + ", ".join(t["tag"] for t in tags)
                    if tags
                    else "轨道近乎静音",
                )
            whole = self.tag_audio(path)
        report = dict(
            input=str(path.resolve()),
            duration_seconds=info.duration,
            device=self.device,
            tracks=tracks,
            whole_tags=whole,
            description="全曲候选语义："
            + ", ".join(t["tag"] for t in whole)
            + "。标签按相似度排序，不代表已校准的出现概率。",
            elapsed_seconds=round(time.perf_counter() - start, 3),
        )
        result = folder / "analysis.json"
        result.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["result_path"] = str(result)
        return report

    def create(
        self, analysis, request="生成混音建议、短视频文案和场景推荐", max_tokens=256
    ):
        """Qwen 只接收明确的逐轨语义证据，不假装听过波形。"""
        if not analysis or "tracks" not in analysis:
            raise ValueError("请先分析音频")
        if request is None or not str(request).strip():
            raise ValueError("请输入创作要求")
        format_creation(analysis, "")  # 模型加载前检查四轨结构。
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self.lock:
            if self.llm is None:
                name = "Qwen/Qwen2-1.5B-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name, local_files_only=True
                )
                self.llm = (
                    AutoModelForCausalLM.from_pretrained(
                        name,
                        local_files_only=True,
                        torch_dtype=torch.float32
                        if self.device == "cpu"
                        else torch.float16,
                    )
                    .to(self.device)
                    .eval()
                )
            evidence = {
                s: {"candidate_tags": [t["tag"] for t in v["tags"]]}
                for s, v in analysis["tracks"].items()
            }
            prompt = (
                "以下是音频分轨分析的候选标签，不是已确认事实："
                + json.dumps(evidence, ensure_ascii=False)
                + "\n需求："
                + str(request)[:1500]
                + '\n只输出JSON：{"caption":"中文短视频配乐文案，20至50字","scenes":["场景一","场景二","场景三"]}。'
                + "三个场景各2至12字。文案写画面故事，不写混音、教学、课程、观众或技术操作。"
                + "不得断言乐器、音质、节拍或处理事实。只推荐可以尝试的生活或视频配乐场景。"
            )
            text = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "你是音乐创作助手。只依据提供的分析给出可执行的建议。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.llm.generate(
                    **batch, max_new_tokens=max_tokens, do_sample=False
                )
            raw = self.tokenizer.decode(
                out[0, batch.input_ids.shape[1] :], skip_special_tokens=True
            )
            result = format_creation(analysis, raw)
            record = dict(
                input=analysis.get("input"),
                request=str(request),
                evidence=evidence,
                raw_model_output=raw,
                **result,
            )
            (self.output / f"creation_{uuid.uuid4().hex}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result["text"]
