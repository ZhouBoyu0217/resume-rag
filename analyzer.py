"""
简历深度分析：摘要、综合评价、JD 匹配。

为什么单独一个文件？
- summary_core 管的是"跨简历的全局问题"（Map-Reduce）
- 这里管的是"单份简历的体检报告"
两者都是调大模型，但提示词、输出结构完全不同，混在一起会很乱。

设计要点：
- 所有函数只吃"原文文本"，不碰 Chroma —— 调用方负责取原文
- 大模型返回统一要求 JSON，解析失败要有兜底，不能让接口崩
"""
import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-v4-pro"


def _ask(system, user, json_mode=True):
    """统一的大模型调用 + JSON 解析容错。

    模型有时会包 ```json 围栏、有时会加一段解释文字。
    这里先剥围栏，再尝试直接解析，最后用正则抠第一个 {...}。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,     # 分析类任务要稳定，不要发挥
    )
    raw = resp.choices[0].message.content.strip()

    if not json_mode:
        return raw

    text = raw
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"_raw": raw, "_parse_failed": True}


# ---------------- 摘要（去掉 Markdown 标题，压成一段） ----------------
def summarize(text):
    """把简历压成 150-260 字的一段话，用于侧边抽屉的“概要”视图。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "把这份简历压缩成 150-260 字的一段话。必须覆盖：姓名、工作/项目年限、"
                    "核心技术栈、最有分量的 1-2 个项目及其量化结果、求职方向。"
                    "只输出正文，不要标题、不要分点、不要寒暄。"
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------------- 综合评价 ----------------
SCORE_SYSTEM = """你是资深的技术招聘面试官。给一份简历按四个维度打分（0-100），并给出具体依据。

打分维度：
1. tech（技术栈完整度）：是否覆盖岗位核心技能，深度如何（只会调 API vs 懂原理）
2. project（项目深度）：项目复杂度、是否从 0 到 1、有没有架构设计
3. impact（量化成果）：是否有真实指标（准确率、耗时、成本、QPS 等）
4. clarity（表达质量）：结构是否清晰、重点是否突出、有无废话

严格要求：
- 分数要有区分度，不要全都给 80 分。平庸的简历就该低分。
- evidence 必须引用简历里的原话或事实，不能空泛地说"经验丰富"。
- 只输出 JSON，不要任何解释文字。

输出格式：
{
  "total": 综合分(0-100，四项加权平均),
  "dimensions": {
    "tech":    {"score": 0-100, "evidence": "一句话依据"},
    "project": {"score": 0-100, "evidence": "一句话依据"},
    "impact":  {"score": 0-100, "evidence": "一句话依据"},
    "clarity": {"score": 0-100, "evidence": "一句话依据"}
  },
  "strengths": ["优点1", "优点2", "优点3"],
  "weaknesses": ["短板1", "短板2", "短板3"],
  "suggestions": ["具体可执行的改进建议1", "建议2", "建议3"]
}"""


def score_resume(text, target_role="大模型应用开发工程师"):
    """综合评分。target_role 决定按什么岗的标准打分。"""
    user = f"目标岗位：{target_role}\n\n简历原文：\n{text}"
    return _ask(SCORE_SYSTEM, user)


# ---------------- JD 匹配 ----------------
MATCH_SYSTEM = """你是技术招聘的简历筛选助手。把简历和岗位 JD 做逐项对比。

要求：
- 只依据 JD 里明确写了的要求做判断，不要自己脑补岗位需求
- 每条要求标注状态：met（满足）/ partial（部分满足）/ missing（不满足）
- evidence 必须引用简历原话；简历里没有就写"简历未提及"
- match_score 是匹配度百分比（0-100），按"必须项"权重高于"加分项"计算
- 只输出 JSON，不要任何解释文字

输出格式：
{
  "match_score": 0-100,
  "summary": "一段话说明整体匹配情况和最关键的差距",
  "requirements": [
    {"item": "JD 里的要求原文", "status": "met|partial|missing", "evidence": "简历里的对应内容或'简历未提及'"}
  ],
  "must_have_gaps": ["必须项里缺失的内容"],
  "how_to_improve": ["怎么改简历/补什么经历能提高匹配度"]
}"""


def match_resume(text, jd):
    """JD 匹配打分。"""
    user = f"岗位 JD：\n{jd}\n\n简历原文：\n{text}"
    return _ask(MATCH_SYSTEM, user)
