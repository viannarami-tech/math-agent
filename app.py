from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import json
import os
import re
import uuid

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI(title="信号与系统 · 认知型多智能体辅导系统")

ConceptId = str
ErrorType = Literal["concept", "calculation", "symbol", "logic", "prerequisite", "method", "transform"]
ResourceType = Literal[
    "lecture", "exercise", "mindmap", "reading", "flashcard", "variant",
    "case", "animation_script", "code_lab", "exam_strategy", "formula_sheet"
]

# =========================
# 1. 信号与系统知识图谱
# =========================

KNOWLEDGE_GRAPH: Dict[str, Dict[str, Any]] = {
    "sig-1.1": {"name": "连续时间与离散时间信号", "pre": [], "level": "L0"},
    "sig-1.2": {"name": "基本信号：冲激、阶跃、指数、正弦", "pre": ["sig-1.1"], "level": "L0"},
    "sys-2.1": {"name": "系统性质：线性、时不变、因果、稳定", "pre": ["sig-1.1"], "level": "L1"},
    "lti-3.1": {"name": "LTI系统与冲激响应", "pre": ["sys-2.1", "sig-1.2"], "level": "L1"},
    "conv-3.2": {"name": "连续时间卷积", "pre": ["lti-3.1"], "level": "L1"},
    "conv-3.3": {"name": "离散时间卷积", "pre": ["lti-3.1"], "level": "L1"},
    "fs-4.1": {"name": "傅里叶级数", "pre": ["sig-1.2"], "level": "L2"},
    "ft-4.2": {"name": "傅里叶变换", "pre": ["conv-3.2"], "level": "L2"},
    "freq-4.3": {"name": "频率响应与滤波", "pre": ["ft-4.2", "lti-3.1"], "level": "L2"},
    "lap-5.1": {"name": "拉普拉斯变换与系统函数", "pre": ["conv-3.2"], "level": "L2"},
    "z-6.1": {"name": "Z变换与离散系统", "pre": ["conv-3.3"], "level": "L2"},
    "sample-7.1": {"name": "采样定理与重建", "pre": ["ft-4.2"], "level": "L2"},
    "state-8.1": {"name": "系统综合分析与考试策略", "pre": ["freq-4.3", "lap-5.1", "z-6.1"], "level": "L3"},
}

# =========================
# 2. 数据结构
# =========================

@dataclass
class BlindSpot:
    concept_id: ConceptId
    concept_name: str
    error_type: ErrorType
    confidence: float
    evidence: str
    root_concept: str

@dataclass
class Profile:
    knowledge_mastery: Dict[str, float] = field(default_factory=lambda: {cid: 0.25 for cid in KNOWLEDGE_GRAPH})
    blind_spots: List[BlindSpot] = field(default_factory=list)
    cognitive_style: str = "exploratory"
    response_style: str = "cautious"
    resource_preference: str = "interactive"
    major: str = "未填写"
    learning_goal: str = "掌握信号与系统核心概念并能解题"
    progress_level: str = "L1"
    error_tendency: Dict[str, int] = field(default_factory=lambda: {
        "concept": 0, "calculation": 0, "symbol": 0, "logic": 0,
        "prerequisite": 0, "method": 0, "transform": 0
    })
    weak_points: List[str] = field(default_factory=list)
    preferred_method: str = "step_by_step"
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

@dataclass
class SessionState:
    session_id: str
    profile: Profile = field(default_factory=Profile)
    messages: List[Dict[str, str]] = field(default_factory=list)
    long_memory: List[Dict[str, Any]] = field(default_factory=list)
    last_assessment: Dict[str, Any] = field(default_factory=dict)
    resource_queue: List[Dict[str, Any]] = field(default_factory=list)
    path: Dict[str, Any] = field(default_factory=dict)
    last_plan: Dict[str, Any] = field(default_factory=dict)

SESSIONS: Dict[str, SessionState] = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)
    major: Optional[str] = None
    goal: Optional[str] = None
    model: Optional[str] = Field(default="auto", description="auto / openai / doubao / local")
    mode: Optional[str] = Field(default="coach", description="coach / solve / exam / diagnose")

# =========================
# 3. 工具函数
# =========================

def get_session(session_id: Optional[str]) -> SessionState:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    sid = str(uuid.uuid4())[:8]
    SESSIONS[sid] = SessionState(session_id=sid)
    return SESSIONS[sid]

def clamp(x: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        text = text.strip()
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text)
    except Exception:
        return fallback

def detect_major_and_preference(text: str, profile: Profile) -> None:
    majors = ["计算机", "人工智能", "软件", "电子", "自动化", "通信", "电气", "机械", "物理", "数学", "考研"]
    for m in majors:
        if m in text:
            profile.major = m
    if any(k in text for k in ["图", "画", "可视化", "波形", "动画", "频谱"]):
        profile.resource_preference = "visual"
    elif any(k in text for k in ["推导", "证明", "公式", "严格"]):
        profile.resource_preference = "textual"
    elif any(k in text for k in ["练习", "题", "互动", "一步步", "带我"]):
        profile.resource_preference = "interactive"
    if any(k in text for k in ["不确定", "可能", "好像", "应该"]):
        profile.response_style = "cautious"
    elif any(k in text for k in ["直接", "给答案", "快点"]):
        profile.response_style = "direct"
    else:
        profile.response_style = "exploratory"

def compact_memory(state: SessionState, max_items: int = 6) -> str:
    if not state.long_memory:
        return "暂无长期记忆。"
    items = state.long_memory[-max_items:]
    return "\n".join([
        f"- 问题：{m.get('question','')}；薄弱点：{m.get('weak_point','未知')}；建议：{m.get('next_action','继续诊断')}"
        for m in items
    ])

def concept_name(cid: str) -> str:
    return KNOWLEDGE_GRAPH.get(cid, {}).get("name", cid)

# =========================
# 4. 多模型路由器：GPT / 豆包 / 本地兜底
# =========================

class LLMRouter:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.doubao_api_key = os.getenv("DOUBAO_API_KEY", "").strip()
        self.doubao_endpoint = os.getenv("DOUBAO_ENDPOINT", "").strip()
        self.doubao_model = os.getenv("DOUBAO_MODEL", self.doubao_endpoint).strip()

    def available(self) -> Dict[str, bool]:
        return {
            "openai": bool(self.openai_api_key and OpenAI),
            "doubao": bool(self.doubao_api_key and self.doubao_model),
            "local": True,
        }

    def call(self, messages: List[Dict[str, str]], model: str = "auto", temperature: float = 0.35) -> str:
        if model == "auto":
            if self.available()["openai"]:
                model = "openai"
            elif self.available()["doubao"]:
                model = "doubao"
            else:
                model = "local"
        if model == "openai" and self.available()["openai"]:
            return self.call_openai(messages, temperature)
        if model == "doubao" and self.available()["doubao"]:
            return self.call_doubao(messages, temperature)
        return self.local_fallback(messages)

    def call_openai(self, messages: List[Dict[str, str]], temperature: float) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def call_doubao(self, messages: List[Dict[str, str]], temperature: float) -> str:
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.doubao_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.doubao_model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def local_fallback(self, messages: List[Dict[str, str]]) -> str:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        cid = heuristic_target_concept(user)
        name = concept_name(cid)
        return (
            f"当前未检测到可用大模型 API，系统进入本地规则兜底模式。\n\n"
            f"【题型判断】该问题初步关联：{name}\n"
            f"【学习建议】请先明确：输入信号 x(t)/x[n]、系统冲激响应 h(t)/h[n]、要求的是时域输出还是频域表达。\n"
            f"【下一步】如果是卷积题，先写 y = x*h；如果是傅里叶题，先判断周期/非周期，再选择傅里叶级数或傅里叶变换。"
        )

llm = LLMRouter()

SYSTEM_PROMPT = """
你是一个认知型“信号与系统”专业课辅导智能体，不是高等数学老师。

你的目标不是只给答案，而是像优秀老师一样：理解学生、规划解法、比较策略、指出盲区、生成个性化训练。

必须遵守：
1. 严格围绕信号与系统课程：信号、系统性质、LTI、卷积、傅里叶级数、傅里叶变换、拉普拉斯变换、Z变换、采样、频率响应。
2. 优先使用信号与系统方法，而不是泛泛的高数技巧。
3. 输出时展示“教学化推理过程”，但不要暴露冗长内部思维；用清晰步骤说明理由。
4. 至少在复杂题中比较两种方法，例如：时域法 / 频域法 / 图解法 / 变换法。
5. 必须包含自我检查：检查单位、定义域、收敛域、边界条件、是否遗漏分段。
6. 如果题目信息不足，先给出需要补充的信息，同时提供一个通用解题框架。
7. 风格：循循善诱，适合中国大学“信号与系统”课程和考研复习。
""".strip()

# =========================
# 5. 规则诊断：兜底 + 辅助 LLM
# =========================

def heuristic_target_concept(text: str) -> str:
    t = text.lower()
    if any(k in text for k in ["离散卷积", "卷积和", "x[n]", "h[n]", "序列"]) or "convolution sum" in t:
        return "conv-3.3"
    if any(k in text for k in ["卷积", "卷积积分", "x(t)", "h(t)"]) or "convolution" in t:
        return "conv-3.2"
    if any(k in text for k in ["傅里叶级数", "FS", "周期信号"]):
        return "fs-4.1"
    if any(k in text for k in ["傅里叶", "频谱", "频域", "FT", "频率响应"]):
        return "ft-4.2"
    if any(k in text for k in ["滤波", "低通", "高通", "带通", "系统函数", "H(jw)", "H(jω)"]):
        return "freq-4.3"
    if any(k in text for k in ["拉普拉斯", "s域", "ROC", "零极点", "系统函数"]):
        return "lap-5.1"
    if any(k in text for k in ["Z变换", "z变换", "z域", "H(z)"]):
        return "z-6.1"
    if any(k in text for k in ["采样", "奈奎斯特", "混叠", "重建"]):
        return "sample-7.1"
    if any(k in text for k in ["线性", "时不变", "因果", "稳定", "BIBO"]):
        return "sys-2.1"
    if any(k in text for k in ["冲激", "阶跃", "指数", "正弦", "单位样值"]):
        return "sig-1.2"
    return "sig-1.1"

class CognitivePlannerAgent:
    def run(self, state: SessionState, text: str, model: str, mode: str) -> Dict[str, Any]:
        fallback_cid = heuristic_target_concept(text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n你现在是 Planner。请只输出 JSON，不要输出 Markdown。"},
            {"role": "user", "content": f"""
学生画像：{json.dumps(serialize_profile(state.profile), ensure_ascii=False)}
长期记忆：{compact_memory(state)}
模式：{mode}
学生输入：{text}

请输出 JSON：
{{
  "problem_type": "题型",
  "target_concept": "必须从这些ID中选择：{list(KNOWLEDGE_GRAPH.keys())}",
  "subtasks": ["子任务1", "子任务2"],
  "methods": [
    {{"name": "方法名", "why": "为什么适用", "risk": "易错点"}}
  ],
  "missing_info": ["缺失信息"],
  "teaching_strategy": "教学策略",
  "difficulty": "L0/L1/L2/L3"
}}
"""}
        ]
        raw = llm.call(messages, model=model, temperature=0.2)
        plan = safe_json_loads(raw, fallback={})
        if not isinstance(plan, dict):
            plan = {}
        plan.setdefault("problem_type", concept_name(fallback_cid))
        plan.setdefault("target_concept", fallback_cid)
        if plan.get("target_concept") not in KNOWLEDGE_GRAPH:
            plan["target_concept"] = fallback_cid
        plan.setdefault("subtasks", ["识别信号与系统问题类型", "选择合适的时域或变换域方法", "逐步求解并检查条件"])
        plan.setdefault("methods", [
            {"name": "时域法", "why": "适合理解卷积、冲激响应和系统性质", "risk": "分段边界容易错"},
            {"name": "变换域法", "why": "适合复杂卷积和系统函数分析", "risk": "容易忽略收敛域或适用条件"},
        ])
        plan.setdefault("missing_info", [])
        plan.setdefault("teaching_strategy", "先判断题型，再给公式，再分步骤推导，最后做自检。")
        plan.setdefault("difficulty", KNOWLEDGE_GRAPH[plan["target_concept"]]["level"])
        return plan

class MultiSolverAgent:
    def run(self, text: str, plan: Dict[str, Any], profile: Profile, model: str) -> List[Dict[str, str]]:
        methods = plan.get("methods", [])[:3] or [{"name": "标准法", "why": "通用", "risk": "无"}]
        outputs = []
        for m in methods:
            method_name = m.get("name", "标准法") if isinstance(m, dict) else str(m)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + f"\n你现在是 Solver，请使用【{method_name}】解决或讲解。"},
                {"role": "user", "content": f"""
学生专业：{profile.major}
学习目标：{profile.learning_goal}
题型计划：{json.dumps(plan, ensure_ascii=False)}
学生问题：{text}

要求：
1. 先说明该方法适用原因。
2. 给出核心公式。
3. 分步骤讲解。
4. 如果信息不足，给出通用框架，不要编造题目条件。
"""}
            ]
            ans = llm.call(messages, model=model, temperature=0.35)
            outputs.append({"method": method_name, "answer": ans})
        return outputs

class VotingAgent:
    def run(self, text: str, plan: Dict[str, Any], solutions: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n你现在是 Voting Judge。请比较多种解法，选出最适合学生的一种。输出 JSON。"},
            {"role": "user", "content": f"""
学生问题：{text}
计划：{json.dumps(plan, ensure_ascii=False)}
候选解法：{json.dumps(solutions, ensure_ascii=False)}

请输出 JSON：
{{
  "best_method": "最佳方法名",
  "reason": "选择理由",
  "merged_answer": "融合后的教学答案",
  "risks": ["仍需注意的风险"]
}}
"""}
        ]
        raw = llm.call(messages, model=model, temperature=0.2)
        voted = safe_json_loads(raw, fallback={})
        if not isinstance(voted, dict):
            voted = {}
        voted.setdefault("best_method", solutions[0]["method"] if solutions else "标准法")
        voted.setdefault("reason", "默认选择第一种完整解法。")
        voted.setdefault("merged_answer", solutions[0]["answer"] if solutions else "当前无法生成完整解答。")
        voted.setdefault("risks", ["注意题目条件是否完整。"])
        return voted

class CriticReflectionAgent:
    def run(self, text: str, plan: Dict[str, Any], voted: Dict[str, Any], model: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n你现在是 Critic + Reflection。检查答案并给出教学反思。只输出 JSON。"},
            {"role": "user", "content": f"""
学生问题：{text}
计划：{json.dumps(plan, ensure_ascii=False)}
当前答案：{voted.get('merged_answer','')}

请输出 JSON：
{{
  "is_reliable": true,
  "corrections": ["需要修正的点"],
  "self_check": ["自检项"],
  "student_likely_confused": ["学生可能困惑点"],
  "next_question": "下一步苏格拉底式追问",
  "weak_point": "本轮暴露的薄弱点",
  "next_action": "下一步学习建议"
}}
"""}
        ]
        raw = llm.call(messages, model=model, temperature=0.2)
        ref = safe_json_loads(raw, fallback={})
        if not isinstance(ref, dict):
            ref = {}
        ref.setdefault("is_reliable", True)
        ref.setdefault("corrections", [])
        ref.setdefault("self_check", ["是否选对时域/频域方法", "是否检查收敛域或边界条件", "是否遗漏分段"])
        ref.setdefault("student_likely_confused", [])
        ref.setdefault("next_question", "你能说说这道题为什么优先用这个方法吗？")
        ref.setdefault("weak_point", concept_name(plan.get("target_concept", "sig-1.1")))
        ref.setdefault("next_action", "完成一道同类型变式题，并复述解题框架。")
        return ref

class FinalResponseAgent:
    def run(self, text: str, plan: Dict[str, Any], voted: Dict[str, Any], reflection: Dict[str, Any], profile: Profile, model: str, mode: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n你现在是最终授课老师。输出给学生看的答案，结构清楚，不要说自己调用了多个Agent。"},
            {"role": "user", "content": f"""
学生问题：{text}
学生画像：专业={profile.major}，目标={profile.learning_goal}，水平={profile.progress_level}，偏好={profile.resource_preference}
模式：{mode}
规划：{json.dumps(plan, ensure_ascii=False)}
最佳解法：{json.dumps(voted, ensure_ascii=False)}
反思与自检：{json.dumps(reflection, ensure_ascii=False)}

请按下面结构输出：
① 题型判断
② 推荐方法与原因
③ 核心公式
④ 分步骤讲解
⑤ 自我检查
⑥ 下一步追问/练习

注意：如果题目信息不足，要明确指出缺什么，并给出通用模板。
"""}
        ]
        return llm.call(messages, model=model, temperature=0.35)

class DiagnosticianAgent:
    def run(self, text: str, profile: Profile, plan: Dict[str, Any]) -> Dict[str, Any]:
        cid = plan.get("target_concept") or heuristic_target_concept(text)
        if cid not in KNOWLEDGE_GRAPH:
            cid = heuristic_target_concept(text)
        blind: List[BlindSpot] = []
        mastered: List[str] = []
        if any(k in text for k in ["不知道", "不懂", "不会", "完全不会", "看不懂"]):
            pre = KNOWLEDGE_GRAPH[cid]["pre"]
            blind.append(BlindSpot(cid, concept_name(cid), "concept", .88, "学生明确表达不会或不懂", pre[0] if pre else cid))
        if any(k in text for k in ["直接相乘", "卷积就是乘", "时域相乘"]):
            blind.append(BlindSpot(cid, concept_name(cid), "concept", .84, "混淆时域卷积与乘法关系", "lti-3.1"))
        if any(k in text for k in ["ROC", "收敛域", "极点"]):
            mastered.append(cid)
        if any(k in text for k in ["分段", "翻转", "平移", "重叠"]):
            mastered.append("conv-3.2" if cid == "conv-3.2" else cid)
        return {
            "target_concept": cid,
            "blind_spots": [asdict(b) for b in blind],
            "mastered_concepts": list(set(mastered)),
            "summary": "已根据学生输入、LLM规划和规则诊断定位知识节点。",
        }

class AssessorAgent:
    def run(self, text: str, diagnosis: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        score = 0.55
        correct = False
        if diagnosis.get("mastered_concepts"):
            score += 0.22
            correct = True
        if diagnosis.get("blind_spots"):
            score -= 0.25
            correct = False
            for b in diagnosis["blind_spots"]:
                errors.append({"type": b["error_type"], "description": b["evidence"], "related_concept": b["concept_id"]})
        if not reflection.get("is_reliable", True):
            score -= 0.08
            errors.append({"type": "logic", "description": "答案可靠性需要进一步检查", "related_concept": diagnosis["target_concept"]})
        if len(text) < 8:
            score -= 0.08
            errors.append({"type": "concept", "description": "输入过短，诊断证据不足", "related_concept": diagnosis["target_concept"]})
        rec = "pass" if score >= .78 else "extra_coaching" if score >= .55 else "remediate" if score >= .35 else "backtrack"
        return {
            "correct": correct,
            "score": round(clamp(score), 2),
            "error_patterns": errors,
            "recommendation": rec,
            "target_concept": diagnosis["target_concept"],
            "summary": "评估完成：{}，建议 {}。".format("基本掌握" if correct else "仍有盲区", rec),
        }

class ProfileBuilderAgent:
    def run(self, state: SessionState, student_text: str, assessment: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
        p = state.profile
        detect_major_and_preference(student_text, p)
        cid = assessment.get("target_concept")
        if cid in KNOWLEDGE_GRAPH:
            delta = 0.10 if assessment.get("correct") else -0.06
            p.knowledge_mastery[cid] = clamp(p.knowledge_mastery.get(cid, 0.25) + delta)
        for e in assessment.get("error_patterns", []):
            et = e.get("type", "concept")
            p.error_tendency[et] = p.error_tendency.get(et, 0) + 1
        if reflection.get("weak_point"):
            wp = reflection["weak_point"]
            if wp not in p.weak_points:
                p.weak_points.append(wp)
            p.weak_points = p.weak_points[-8:]
        p.blind_spots = [BlindSpot(**b) if isinstance(b, dict) else b for b in assessment.get("blind_spots", [])]
        scores = p.knowledge_mastery
        avg = sum(scores.values()) / len(scores)
        p.progress_level = "L0" if avg < .25 else "L1" if avg < .5 else "L2" if avg < .75 else "L3"
        p.updated_at = datetime.now().isoformat(timespec="seconds")
        return {
            "knowledge_mastery": [{"concept_id": k, "name": concept_name(k), "score": round(v, 2)} for k, v in scores.items()],
            "dimensions": {
                "知识掌握": round(avg, 2),
                "认知风格": p.cognitive_style,
                "易错倾向": max(p.error_tendency, key=p.error_tendency.get),
                "专业背景": p.major,
                "学习目标": p.learning_goal,
                "资源偏好": p.resource_preference,
                "学习进度": p.progress_level,
                "薄弱点": "、".join(p.weak_points[-3:]) if p.weak_points else "暂无",
            }
        }

class ResourceGeneratorAgent:
    def run(self, profile: Profile, diagnosis: Dict[str, Any], assessment: Dict[str, Any], reflection: Dict[str, Any]) -> List[Dict[str, Any]]:
        cid = assessment.get("target_concept", diagnosis.get("target_concept", "sig-1.1"))
        name = concept_name(cid)
        return [
            {"type": "lecture", "title": f"{name} 结构化讲义", "content": f"从物理直觉、数学定义、典型题型、易错点四层讲解 {name}。"},
            {"type": "formula_sheet", "title": f"{name} 公式卡", "content": "整理核心公式、适用条件、常见符号和考试陷阱。"},
            {"type": "exercise", "title": f"{name} 分层练习", "content": "基础 2 题、提高 2 题、综合 1 题，每题附提示。"},
            {"type": "mindmap", "title": f"{name} 知识图谱", "content": f"{name} → 前置概念 → 解题方法 → 自检项 → 变式题。"},
            {"type": "variant", "title": "错题变式", "content": f"围绕薄弱点“{reflection.get('weak_point', name)}”生成同构变式，训练迁移。"},
            {"type": "exam_strategy", "title": "考研/期末策略", "content": "先判题型，再选域：卷积题优先考虑变换域；系统性质题优先检查定义。"},
            {"type": "animation_script", "title": "波形动画脚本", "content": "用翻转、平移、重叠面积解释卷积；用频谱移动解释调制和采样。"},
            {"type": "code_lab", "title": "Python 数值实验", "content": "用 numpy 计算离散卷积，用 matplotlib 画信号和频谱，验证理论结果。"},
        ]

class PlannerPathAgent:
    def run(self, profile: Profile, assessment: Dict[str, Any], resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        rec = assessment.get("recommendation", "extra_coaching")
        cid = assessment.get("target_concept", "sig-1.1")
        if rec == "pass":
            steps = ["完成 1 道迁移题", "尝试另一种解法", "进入下一知识点"]
            queue = ["variant", "exam_strategy", "mindmap"]
        elif rec == "extra_coaching":
            steps = ["回答 1 个追问", "复述核心公式适用条件", "做 2 道同类题"]
            queue = ["formula_sheet", "exercise", "variant"]
        elif rec == "remediate":
            steps = ["回看结构化讲义", "做基础题", "重新诊断薄弱点"]
            queue = ["lecture", "formula_sheet", "exercise"]
        else:
            steps = ["回到前置概念", "用图像/动画建立直觉", "再尝试当前题型"]
            queue = ["lecture", "animation_script", "mindmap"]
        return {
            "current_level": profile.progress_level,
            "target_concept": cid,
            "target_name": concept_name(cid),
            "next_steps": steps,
            "resource_queue": [r for r in resources if r["type"] in queue],
            "promotion_condition": "连续 2 轮 score ≥ 0.8，且能说清方法选择理由与自检项。",
            "adjustment_strategy": "根据下一轮回答动态调整方法、难度和资源类型。",
        }

class QualityGateAgent:
    def run(self, final_answer: str, resources: List[Dict[str, Any]], reflection: Dict[str, Any]) -> Dict[str, Any]:
        issues = []
        if len(resources) < 5:
            issues.append({"severity": "error", "description": "资源类型不足 5 类", "suggestion": "补充讲义、练习、公式卡、变式、代码实验"})
        if not final_answer or len(final_answer) < 40:
            issues.append({"severity": "error", "description": "最终回答过短", "suggestion": "补充题型判断、公式和步骤"})
        if not reflection.get("self_check"):
            issues.append({"severity": "warning", "description": "缺少自检项", "suggestion": "补充边界、ROC、分段、单位等检查"})
        return {"passed": not any(i["severity"] == "error" for i in issues), "issues": issues}

# =========================
# 6. 序列化
# =========================

def serialize_profile(p: Profile) -> Dict[str, Any]:
    d = asdict(p)
    d["blind_spots"] = [asdict(b) if not isinstance(b, dict) else b for b in p.blind_spots]
    d["knowledge_mastery_named"] = [
        {"concept_id": cid, "name": concept_name(cid), "score": round(score, 2)}
        for cid, score in p.knowledge_mastery.items()
    ]
    return d

# 实例化智能体
cognitive_planner = CognitivePlannerAgent()
multi_solver = MultiSolverAgent()
voting_agent = VotingAgent()
critic_reflector = CriticReflectionAgent()
final_responder = FinalResponseAgent()
diagnostician = DiagnosticianAgent()
assessor = AssessorAgent()
profile_builder = ProfileBuilderAgent()
resource_gen = ResourceGeneratorAgent()
path_planner = PlannerPathAgent()
quality_gate = QualityGateAgent()

# =========================
# 7. API
# =========================

@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "title": "信号与系统 · 认知型多智能体辅导系统",
        "llm_available": llm.available(),
        "knowledge_nodes": len(KNOWLEDGE_GRAPH),
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    state = get_session(req.session_id)
    if req.major:
        state.profile.major = req.major
    if req.goal:
        state.profile.learning_goal = req.goal

    user_model = req.model or "auto"
    mode = req.mode or "coach"
    state.messages.append({"role": "student", "content": req.message})

    try:
        plan = cognitive_planner.run(state, req.message, user_model, mode)
        solutions = multi_solver.run(req.message, plan, state.profile, user_model)
        voted = voting_agent.run(req.message, plan, solutions, user_model)
        reflection = critic_reflector.run(req.message, plan, voted, user_model)
        final_answer = final_responder.run(req.message, plan, voted, reflection, state.profile, user_model, mode)
    except Exception as e:
        cid = heuristic_target_concept(req.message)
        plan = {
            "problem_type": concept_name(cid),
            "target_concept": cid,
            "subtasks": ["识别题型", "列公式", "分步骤求解", "检查条件"],
            "methods": [
                {"name": "时域法", "why": "适合理解系统输入输出关系", "risk": "分段边界易错"},
                {"name": "变换域法", "why": "适合卷积和系统函数", "risk": "注意收敛域"},
            ],
            "missing_info": [],
            "teaching_strategy": "规则兜底教学",
            "difficulty": KNOWLEDGE_GRAPH[cid]["level"],
            "llm_error": str(e),
        }
        solutions = []
        voted = {"best_method": "规则兜底", "reason": "大模型调用失败，使用本地诊断。", "merged_answer": "", "risks": [str(e)]}
        reflection = {
            "is_reliable": False,
            "corrections": ["大模型调用失败，已切换兜底模式。"],
            "self_check": ["检查题目条件是否完整", "检查是否应使用卷积或变换"],
            "student_likely_confused": [concept_name(cid)],
            "next_question": "请补充题目中的 x(t)/x[n]、h(t)/h[n] 或系统表达式。",
            "weak_point": concept_name(cid),
            "next_action": "先完成题型识别和公式选择。",
        }
        final_answer = llm.local_fallback([{"role": "user", "content": req.message}])

    diagnosis = diagnostician.run(req.message, state.profile, plan)
    assessment = assessor.run(req.message, diagnosis, reflection)
    # 把诊断盲区传入 assessment，便于画像更新和前端展示
    assessment_with_blind = {**assessment, "blind_spots": diagnosis.get("blind_spots", [])}
    profile_delta = profile_builder.run(state, req.message, assessment_with_blind, reflection)
    resources = resource_gen.run(state.profile, diagnosis, assessment, reflection)
    path = path_planner.run(state.profile, assessment, resources)
    quality = quality_gate.run(final_answer, resources, reflection)

    memory_item = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": req.message,
        "target_concept": plan.get("target_concept"),
        "weak_point": reflection.get("weak_point"),
        "next_action": reflection.get("next_action"),
        "score": assessment.get("score"),
    }
    state.long_memory.append(memory_item)
    state.long_memory = state.long_memory[-30:]

    state.last_assessment = assessment
    state.resource_queue = path["resource_queue"]
    state.path = path
    state.last_plan = plan
    state.messages.append({"role": "coach", "content": final_answer})

    return {
        "session_id": state.session_id,
        "reply": final_answer,
        "profile": serialize_profile(state.profile),
        "profile_delta": profile_delta,
        "plan": plan,
        "solutions": solutions,
        "voted": voted,
        "reflection": reflection,
        "diagnosis": diagnosis,
        "assessment": assessment,
        "resources": resources,
        "path": path,
        "quality": quality,
        "llm_available": llm.available(),
    }

@app.get("/api/profile/{session_id}")
def get_profile(session_id: str):
    state = SESSIONS.get(session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return serialize_profile(state.profile)

@app.get("/api/export/{session_id}")
def export_report(session_id: str):
    state = SESSIONS.get(session_id)
    if not state:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "session_id": session_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": serialize_profile(state.profile),
        "messages": state.messages,
        "long_memory": state.long_memory,
        "last_plan": state.last_plan,
        "last_assessment": state.last_assessment,
        "path": state.path,
        "resource_queue": state.resource_queue,
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

# =========================
# 8. 前端页面
# =========================

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>信号与系统 · 认知型多智能体辅导系统</title>
<style>
:root{--bg:#0b1020;--card:#121a33;--muted:#90a4c3;--line:#253250;--text:#eaf1ff;--brand:#67e8f9;--brand2:#a78bfa;--ok:#34d399;--warn:#fbbf24;--bad:#fb7185}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#1e3a8a55,transparent 35%),linear-gradient(135deg,#08111f,#0b1020 55%,#111827);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",Arial,sans-serif}.wrap{max-width:1320px;margin:auto;padding:28px}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:24px;align-items:center;margin-bottom:24px}.badge{display:inline-flex;gap:8px;border:1px solid #315074;background:#10233d;padding:8px 12px;border-radius:999px;color:#bde7ff;font-size:13px}h1{font-size:46px;margin:16px 0 10px}.grad{background:linear-gradient(90deg,var(--brand),var(--brand2));-webkit-background-clip:text;color:transparent}.sub{color:#bfcae0;font-size:17px;line-height:1.8}.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:20px}.card{background:linear-gradient(180deg,#141d38ee,#10182fee);border:1px solid var(--line);border-radius:24px;box-shadow:0 20px 50px #0005;padding:20px}.features{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}.feature{padding:16px;border-radius:18px;border:1px solid #2a3b5c;background:#0d172c}.feature b{display:block;margin-bottom:8px}.chat{height:470px;overflow:auto;border:1px solid var(--line);border-radius:18px;padding:16px;background:#0a1224}.msg{margin:10px 0;display:flex}.msg.student{justify-content:flex-end}.bubble{max-width:88%;padding:12px 14px;border-radius:16px;line-height:1.7;white-space:pre-wrap}.student .bubble{background:#2563eb}.coach .bubble{background:#1f2937;border:1px solid #374151}.form{display:flex;gap:10px;margin-top:12px}input,textarea,select{background:#0b1327;border:1px solid #30405f;border-radius:14px;color:var(--text);padding:12px;outline:none}textarea{flex:1;min-height:56px;resize:vertical}button{border:0;border-radius:14px;background:linear-gradient(90deg,#06b6d4,#8b5cf6);color:white;font-weight:700;padding:0 18px;cursor:pointer}.meta{display:grid;grid-template-columns:1fr 1fr .55fr .55fr;gap:10px;margin-bottom:12px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.tab{border:1px solid #33415f;background:#0b1327;color:#cfe1ff;border-radius:999px;padding:8px 12px;cursor:pointer}.tab.active{background:#1d4ed8}.panel{display:none}.panel.active{display:block}.krow{display:grid;grid-template-columns:180px 1fr 46px;gap:10px;align-items:center;margin:10px 0;color:#dbeafe}.bar{height:10px;background:#1f2937;border-radius:999px;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,#22d3ee,#a78bfa)}.pill{display:inline-flex;margin:4px 6px 4px 0;padding:6px 10px;border-radius:999px;background:#172554;border:1px solid #315074;color:#c7d2fe;font-size:13px}.resource{border:1px solid #33415f;background:#0c162b;border-radius:16px;padding:13px;margin:10px 0}.resource h4{margin:0 0 6px}.path li{margin:10px 0}.json{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#c4d3ee;max-height:420px;overflow:auto}.footer{color:#7f91b4;text-align:center;margin-top:24px}.radar{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.dim{padding:12px;border:1px solid #33415f;border-radius:14px;background:#0b1327}.dim small{display:block;color:var(--muted);margin-bottom:6px}.small{font-size:12px;color:var(--muted)}@media(max-width:980px){.hero,.grid{grid-template-columns:1fr}.features{grid-template-columns:1fr}.meta{grid-template-columns:1fr}h1{font-size:34px}}
</style>
</head>
<body><div class="wrap">
<section class="hero"><div><div class="badge">GPT / 豆包 / 本地兜底 · 认知型多智能体架构</div><h1><span class="grad">信号与系统</span><br/>认知型智能辅导系统</h1><p class="sub">系统会进行题型识别、任务规划、多解法生成、最佳策略选择、自我检查、学习画像更新和个性化资源推荐。适合期末、考研和专业课补基础。</p><div class="features"><div class="feature"><b>Meta Planner</b><span>自动拆题与选策略</span></div><div class="feature"><b>Multi-Solver</b><span>时域法 / 频域法 / 图解法比较</span></div><div class="feature"><b>Reflection Memory</b><span>反思、记忆、持续进化</span></div></div></div><div class="card"><h3>智能体模块</h3><span class="pill">Planner</span><span class="pill">Multi-Solver</span><span class="pill">Voting</span><span class="pill">Critic</span><span class="pill">Reflection</span><span class="pill">Profile</span><span class="pill">Resources</span><span class="pill">Path</span><p class="sub">覆盖：卷积、LTI系统、傅里叶、拉普拉斯、Z变换、采样、频率响应。</p><p class="small" id="status">正在检测模型状态...</p></div></section>
<section class="grid"><div class="card"><h2>开始学习</h2><div class="meta"><input id="major" placeholder="专业，如：通信 / 电子 / 自动化"/><input id="goal" placeholder="学习目标，如：考研 / 期末 / 补基础"/><select id="model"><option value="auto">自动模型</option><option value="openai">GPT</option><option value="doubao">豆包</option><option value="local">本地兜底</option></select><select id="mode"><option value="coach">辅导模式</option><option value="solve">解题模式</option><option value="exam">考试模式</option><option value="diagnose">诊断模式</option></select></div><div class="chat" id="chat"><div class="msg coach"><div class="bubble">你好，我是信号与系统认知型辅导智能体。你可以直接输入：我不懂卷积，或者发一道傅里叶变换 / 拉普拉斯 / Z变换 / 采样题。我会先判断题型，再给出多种方法并做自检。</div></div></div><div class="form"><textarea id="input" placeholder="例如：为什么 LTI 系统的输出等于 x(t) 和 h(t) 的卷积？或者直接粘贴题目"></textarea><button onclick="send()">发送</button></div></div><div class="card"><div class="tabs"><button class="tab active" onclick="show('profile')">学习画像</button><button class="tab" onclick="show('plan')">思考规划</button><button class="tab" onclick="show('diagnosis')">诊断评估</button><button class="tab" onclick="show('resources')">资源推送</button><button class="tab" onclick="show('path')">学习路径</button><button class="tab" onclick="show('raw')">JSON</button></div><div id="profile" class="panel active"><h3>画像维度</h3><div class="radar" id="dims"></div><h3>知识掌握</h3><div id="mastery"></div></div><div id="plan" class="panel"><h3>智能体规划</h3><div id="planBox"></div></div><div id="diagnosis" class="panel"><h3>盲区与评估</h3><div id="diag"></div></div><div id="resources" class="panel"><h3>个性化资源</h3><div id="res"></div></div><div id="path" class="panel"><h3>动态学习路径</h3><ol class="path" id="steps"></ol><div id="queue"></div></div><div id="raw" class="panel"><pre class="json" id="json"></pre></div></div></section><div class="footer">© Signal & Systems Cognitive Tutor · Multi-Agent Architecture</div></div>
<script>
let sessionId = localStorage.getItem('ss_session_id') || null;let last=null;
async function loadStatus(){try{const r=await fetch('/api/status');const s=await r.json();document.getElementById('status').textContent='模型状态：GPT '+(s.llm_available.openai?'可用':'未配置')+'｜豆包 '+(s.llm_available.doubao?'可用':'未配置')+'｜本地兜底 可用';}catch(e){document.getElementById('status').textContent='状态检测失败';}}
function addMsg(role, text){const box=document.getElementById('chat');const div=document.createElement('div');div.className='msg '+role;div.innerHTML='<div class="bubble">'+escapeHtml(text)+'</div>';box.appendChild(div);box.scrollTop=box.scrollHeight}
function escapeHtml(s){return String(s).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
async function send(){const input=document.getElementById('input');const text=input.value.trim();if(!text)return;addMsg('student',text);input.value='';addMsg('coach','正在规划解题路径、生成多种解法并自检...');const body={session_id:sessionId,message:text,major:document.getElementById('major').value,goal:document.getElementById('goal').value,model:document.getElementById('model').value,mode:document.getElementById('mode').value};const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();last=data;sessionId=data.session_id;localStorage.setItem('ss_session_id',sessionId);const bubbles=document.querySelectorAll('.msg.coach .bubble');bubbles[bubbles.length-1].textContent=data.reply;render(data)}
function show(id){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));event.target.classList.add('active');document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active')}
function render(data){renderProfile(data.profile,data.profile_delta);renderPlan(data);renderDiagnosis(data);renderResources(data.resources);renderPath(data.path);document.getElementById('json').textContent=JSON.stringify(data,null,2)}
function renderProfile(p,delta){const dims=(delta&&delta.dimensions)||{};document.getElementById('dims').innerHTML=Object.entries(dims).map(([k,v])=>'<div class="dim"><small>'+k+'</small><b>'+escapeHtml(v)+'</b></div>').join('');document.getElementById('mastery').innerHTML=(p.knowledge_mastery_named||[]).map(x=>'<div class="krow"><span>'+escapeHtml(x.name)+'</span><div class="bar"><span style="width:'+(x.score*100)+'%"></span></div><b>'+x.score+'</b></div>').join('')}
function renderPlan(d){const p=d.plan||{};const methods=(p.methods||[]).map(m=>'<div class="resource"><h4>'+escapeHtml(m.name||m)+'</h4><p>适用：'+escapeHtml(m.why||'')+'</p><p>风险：'+escapeHtml(m.risk||'')+'</p></div>').join('');document.getElementById('planBox').innerHTML='<p><b>题型：</b>'+escapeHtml(p.problem_type||'')+'</p><p><b>知识点：</b>'+escapeHtml(p.target_concept||'')+'</p><p><b>教学策略：</b>'+escapeHtml(p.teaching_strategy||'')+'</p><h4>候选方法</h4>'+methods+'<h4>自检</h4><p>'+escapeHtml((d.reflection&&d.reflection.self_check||[]).join('；'))+'</p>'}
function renderDiagnosis(d){const bs=d.diagnosis.blind_spots||[];document.getElementById('diag').innerHTML='<p><b>评分：</b>'+d.assessment.score+' / 1.0　<b>建议：</b>'+d.assessment.recommendation+'</p>'+(bs.length?bs.map(b=>'<div class="resource"><h4>'+escapeHtml(b.concept_name)+' · '+escapeHtml(b.error_type)+'</h4><p>'+escapeHtml(b.evidence)+'</p><small>根因：'+escapeHtml(b.root_concept)+'；置信度：'+b.confidence+'</small></div>').join(''):'<p>暂无明显盲区，可以进入下一层追问。</p>')+'<h4>反思</h4><p>'+escapeHtml((d.reflection.student_likely_confused||[]).join('；')||'暂无')+'</p><h4>质量把关</h4>'+((d.quality.issues||[]).length?d.quality.issues.map(i=>'<p>'+escapeHtml(i.severity)+'：'+escapeHtml(i.description)+'</p>').join(''):'<p>通过。</p>')}
function renderResources(resources){document.getElementById('res').innerHTML=(resources||[]).map(r=>'<div class="resource"><h4>'+label(r.type)+'｜'+escapeHtml(r.title)+'</h4><p>'+escapeHtml(r.content)+'</p></div>').join('')}
function renderPath(path){document.getElementById('steps').innerHTML=(path.next_steps||[]).map(s=>'<li>'+escapeHtml(s)+'</li>').join('');document.getElementById('queue').innerHTML='<h4>推荐资源队列</h4>'+(path.resource_queue||[]).map(r=>'<span class="pill">'+label(r.type)+'</span>').join('')+'<p><b>晋级条件：</b>'+escapeHtml(path.promotion_condition||'')+'</p>'}
function label(t){return {lecture:'讲义',exercise:'练习',mindmap:'思维导图',reading:'拓展阅读',flashcard:'速记卡',variant:'错题变式',case:'专业案例',animation_script:'动画脚本',code_lab:'代码实操',exam_strategy:'考试策略',formula_sheet:'公式卡'}[t]||t}
document.getElementById('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}})
loadStatus();
</script></body></html>'''
