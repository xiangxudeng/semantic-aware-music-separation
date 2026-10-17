"""四板块 Gradio 演示；默认仅在本机提供服务。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from music import MusicSystem


def build(system):
    def analyze(path):
        try:
            report = system.analyze(path)
            return (
                report,
                report["description"],
                *[
                    report["tracks"][s]["path"]
                    for s in ["vocals", "drums", "bass", "other"]
                ],
                report["result_path"],
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def tag(path):
        try:
            return [
                [x["tag"], round(x["similarity"], 4)] for x in system.tag_audio(path)
            ]
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def search(text):
        try:
            rows = system.search(text)
            return [
                [x["clip_id"], x["tags"], round(x["similarity"], 4)] for x in rows
            ], rows[0]["path"]
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def create(report, request):
        try:
            return system.create(report, request)
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="音乐分轨与语义理解", analytics_enabled=False) as demo:
        gr.Markdown(
            "# 音乐分轨与语义理解\n上传音乐，查看四轨、候选标签，按描述搜歌并生成创作建议。CPU 首次加载需要等待。"
        )
        state = gr.State()
        with gr.Tab("分离与全曲理解"):
            audio = gr.Audio(type="filepath", label="输入音乐")
            button = gr.Button("分离并分析", variant="primary")
            description = gr.Textbox(label="全曲语义")
            stems = [
                gr.Audio(label=s, type="filepath")
                for s in ["人声", "鼓", "贝斯", "其他"]
            ]
            file = gr.File(label="下载分析结果")
            button.click(analyze, audio, [state, description, *stems, file])
        with gr.Tab("音乐标签"):
            tag_audio = gr.Audio(type="filepath", label="音频")
            tb = gr.Button("识别候选标签")
            table = gr.Dataframe(headers=["标签", "余弦相似度"], interactive=False)
            tb.click(tag, tag_audio, table)
            gr.Markdown("相似度用于排序，不是标签出现的概率。")
        with gr.Tab("文字搜歌"):
            query = gr.Textbox(
                label="音乐描述", placeholder="例如 A calm piano piece without singing"
            )
            sb = gr.Button("搜索 Top 5")
            results = gr.Dataframe(
                headers=["片段编号", "参考标签", "余弦相似度"], interactive=False
            )
            preview = gr.Audio(label="第一条试听")
            sb.click(search, query, [results, preview])
            gr.Markdown("请优先使用英文描述。当前中文直接查询未通过自由查询评测，系统尚未加入翻译步骤。")
        with gr.Tab("创作助手"):
            gr.Markdown("先在分离板块分析一首音乐，再生成基于逐轨候选标签的建议。")
            gr.Markdown("四轨建议是需要试听确认的操作清单；文案和场景由模型生成并校验，不合格时明确显示规则草稿来源。")
            request = gr.Textbox(
                value="生成混音建议、短视频文案和场景推荐", label="创作需求"
            )
            cb = gr.Button("生成建议")
            answer = gr.Textbox(label="创作建议", lines=14)
            cb.click(create, [state, request], answer)
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    system = MusicSystem(a.assets, a.device)
    build(system).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=a.port,
        share=False,
        allowed_paths=[str(system.output), str(system.assets / "data" / "mtt")],
    )
