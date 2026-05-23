from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import json
import math
import re
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="苏格拉底教练 · 多智能体高数辅导系统")

ConceptId = str
ErrorType = Literal["concept", "calculation", "symbol", "logic", "prerequisite"]
ResourceType = Literal[
    "lecture", "exercise", "mindmap", "reading", "flashcard", "variant", "case", "animation_script", "code_lab"
]

KNOWLEDGE_GRAPH: Dict[str, Dict[str, Any]] = {
    "limit-1.1.1": {"name": "极限直观理解", "pre": [], "level": "L0"},
    "limit-1.1.2": {"name": "ε-N/ε-δ 定义", "pre": ["limit-1.1.1"], "level": "L1"},
    "limit-1.2.1": {"name": "函数极限四则运算", "pre": ["limit-1.1.2"], "level": "L1"},
    "limit-1.3.1": {"name": "无穷小与无穷大", "pre": ["limit-1.1.2"], "level": "L1"},
    "limit-1.3.3": {"name": "等价无穷小替换", "pre": ["limit-1.3.1"], "level": "L2"},
    "limit-1.3.4": {"name": "常见等价公式与适用条件", "pre": ["limit-1.3.3"], "level": "L2"},
    "limit-1.4.1": {"name": "两个重要极限", "pre": ["limit-1.3.3"], "level": "L2"},
    "limit-2.1.1": {"name": "连续性定义", "pre": ["limit-1.1.2"], "level": "L2"},
}

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
    learning_goal: str = "掌握高等数学核心概念并能解题"
    progress_level: str = "L1"
    error_tendency: Dict[str, int] = field(default_factory=lambda: {"concept":0,"calculation":0,"symbol":0,"logic":0,"prerequisite":0})
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

@dataclass
class SessionState:
    session_id: str
    profile: Profile = field(default_factory=Profile)
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_assessment: Dict[str, Any] = field(default_factory=dict)
    resource_queue: List[Dict[str, Any]] = field(default_factory=list)
    path: Dict[str, Any] = field(default_factory=dict)

SESSIONS: Dict[str, SessionState] = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)
    major: Optional[str] = None
    goal: Optional[str] = None


def get_session(session_id: Optional[str]) -> SessionState:
    if not session_id or session_id not in SESSIONS:
        sid = str(uuid.uuid4())[:8]
        SESSIONS[sid] = SessionState(session_id=sid)
    return SESSIONS[session_id] if session_id in SESSIONS else SESSIONS[sid]


def clamp(x: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def detect_major_and_preference(text: str, profile: Profile) -> None:
    majors = ["计算机", "人工智能", "软件", "电子", "机械", "金融", "土木", "物理", "数学", "自动化", "通信", "经济"]
    for m in majors:
        if m in text:
            profile.major = m
    if any(k in text for k in ["图", "画", "可视化", "动画", "视频"]):
        profile.resource_preference = "visual"
    elif any(k in text for k in ["文字", "讲义", "推导", "证明"]):
        profile.resource_preference = "textual"
    elif any(k in text for k in ["练习", "题", "互动", "一步步"]):
        profile.resource_preference = "interactive"
    if any(k in text for k in ["不确定", "可能", "好像", "应该"]):
        profile.response_style = "cautious"
    elif any(k in text for k in ["直接", "就是", "显然"]):
        profile.response_style = "impulsive"
    else:
        profile.response_style = "exploratory"


class ProfileBuilderAgent:
    def run(self, state: SessionState, student_text: str, assessment: Dict[str, Any]) -> Dict[str, Any]:
        p = state.profile
        detect_major_and_preference(student_text, p)
        if assessment.get("target_concept"):
            cid = assessment["target_concept"]
            delta = 0.12 if assessment.get("correct") else -0.08
            p.knowledge_mastery[cid] = clamp(p.knowledge_mastery.get(cid, 0.25) + delta)
        for e in assessment.get("error_patterns", []):
            p.error_tendency[e["type"]] = p.error_tendency.get(e["type"], 0) + 1
        p.blind_spots = [BlindSpot(**b) if isinstance(b, dict) else b for b in assessment.get("blind_spots", [])]
        scores = p.knowledge_mastery
        avg = sum(scores.values()) / len(scores)
        p.progress_level = "L0" if avg < .25 else "L1" if avg < .5 else "L2" if avg < .75 else "L3"
        p.updated_at = datetime.now().isoformat(timespec="seconds")
        return {
            "knowledge_mastery": [{"concept_id": k, "name": KNOWLEDGE_GRAPH[k]["name"], "score": round(v, 2)} for k, v in scores.items()],
            "dimensions": {
                "知识掌握": round(avg, 2),
                "认知风格": p.cognitive_style,
                "易错倾向": max(p.error_tendency, key=p.error_tendency.get),
                "专业背景": p.major,
                "学习目标": p.learning_goal,
                "资源偏好": p.resource_preference,
                "学习进度": p.progress_level,
                "回答风格": p.response_style,
            }
        }


class DiagnosticianAgent:
    def run(self, text: str, profile: Profile) -> Dict[str, Any]:
        lower = text.lower()
        blind: List[BlindSpot] = []
        mastered: List[str] = []
        target = "limit-1.1.1"
        if any(k in text for k in ["等价", "无穷小", "sinx", "tanx", "ln", "e^"]):
            target = "limit-1.3.3"
        elif any(k in text for k in ["连续", "间断"]):
            target = "limit-2.1.1"
        elif any(k in text for k in ["四则", "加减", "乘除"]):
            target = "limit-1.2.1"
        elif any(k in text for k in ["ε", "epsilon", "定义", "N", "delta"]):
            target = "limit-1.1.2"

        if any(k in text for k in ["不知道", "不懂", "不会", "完全不会"]):
            blind.append(BlindSpot(target, KNOWLEDGE_GRAPH[target]["name"], "concept", .88, "学生明确表达不会", KNOWLEDGE_GRAPH[target]["pre"][0] if KNOWLEDGE_GRAPH[target]["pre"] else target))
        if "约掉" in text or "/0" in lower or "分母为0" in text:
            blind.append(BlindSpot("limit-1.2.1", "函数极限四则运算", "logic", .82, "存在把趋近过程当作直接代入或除零处理的风险", "limit-1.1.2"))
        if any(k in text for k in ["sinx=x", "1-cosx", "等价替换"]):
            if not any(k in text for k in ["趋于0", "x->0", "x→0", "局部", "条件"]):
                blind.append(BlindSpot("limit-1.3.4", "常见等价公式与适用条件", "prerequisite", .76, "使用等价无穷小时未说明适用条件", "limit-1.3.3"))
            else:
                mastered.append("limit-1.3.3")
        if re.search(r"\b[0-9]+\s*[+\-*/]\s*[0-9]+\b", text):
            mastered.append(target)
        return {
            "target_concept": target,
            "blind_spots": [asdict(b) for b in blind],
            "mastered_concepts": mastered,
            "summary": "已根据回答定位知识节点与潜在盲区。"
        }


class AssessorAgent:
    def run(self, text: str, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        score = 0.55
        correct = False
        if diagnosis["mastered_concepts"]:
            score += 0.25
            correct = True
        if diagnosis["blind_spots"]:
            score -= 0.25
            correct = False
            for b in diagnosis["blind_spots"]:
                errors.append({"type": b["error_type"], "description": b["evidence"], "related_concept": b["concept_id"]})
        if len(text) < 8:
            score -= 0.1
            errors.append({"type": "concept", "description": "回答过短，证据不足", "related_concept": diagnosis["target_concept"]})
        rec = "pass" if score >= .78 else "extra_coaching" if score >= .55 else "remediate" if score >= .35 else "backtrack"
        return {
            "correct": correct,
            "score": round(clamp(score), 2),
            "error_patterns": errors,
            "recommendation": rec,
            "target_concept": diagnosis["target_concept"],
            "summary": "评估完成：{}，建议 {}。".format("基本掌握" if correct else "仍有盲区", rec)
        }


class SocraticCoachAgent:
    def run(self, text: str, profile: Profile, assessment: Dict[str, Any]) -> Dict[str, Any]:
        cid = assessment.get("target_concept", "limit-1.1.1")
        name = KNOWLEDGE_GRAPH[cid]["name"]
        score = assessment.get("score", .5)
        if "不知道" in text or "不懂" in text or score < .35:
            level = 0
            msg = f"先不急着做题。你能用自己的话说说：{name} 想解决的核心问题是什么吗？"
        elif score < .6:
            level = 1
            msg = f"你刚才的思路有一部分是对的。请补一句：使用这个方法时，变量需要满足什么条件？"
        elif score < .78:
            level = 2
            msg = f"进一步想一想：如果把条件去掉，这个结论还成立吗？请给一个反例或边界情况。"
        else:
            level = 3
            msg = f"很好。现在换个角度：你能把 {name} 和它的前置概念联系起来，说明为什么它不是单独的公式记忆吗？"
        return {
            "level": level,
            "message": msg,
            "target_concept": cid,
            "confidence": round(score, 2),
            "hint": "回答时请写出条件、理由和一个例子。",
            "should_generate_resource": score < .45,
            "should_assess": score > .72
        }


class ResourceGeneratorAgent:
    def run(self, profile: Profile, diagnosis: Dict[str, Any], assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        cid = assessment.get("target_concept", diagnosis.get("target_concept", "limit-1.1.1"))
        name = KNOWLEDGE_GRAPH[cid]["name"]
        major = profile.major
        return [
            {"type": "lecture", "title": f"{name} 结构化讲义", "content": f"先讲直观意义，再讲形式条件，最后给出常见错误。适合当前 {profile.progress_level} 水平。"},
            {"type": "exercise", "title": f"{name} 分层练习", "content": "基础题 2 道、进阶题 2 道、挑战题 1 道，每题附步骤提示。"},
            {"type": "mindmap", "title": f"{name} 知识图谱", "content": f"{name} --> 前置概念 --> 适用条件 --> 典型题型 --> 易错点。"},
            {"type": "reading", "title": f"{major} 专业拓展阅读", "content": f"把 {name} 与 {major} 中的建模、算法收敛或变化率问题连接。"},
            {"type": "flashcard", "title": "速记卡", "content": "正面：什么时候能用该结论？反面：必须检查趋近条件与等价阶。"},
            {"type": "variant", "title": "错题变式", "content": "保留核心结构，改变变量替换、趋近方向和函数组合，训练迁移能力。"},
            {"type": "animation_script", "title": "2 分钟动画脚本", "content": "用点沿曲线靠近目标点的动画解释“趋近但不等于”。"},
            {"type": "code_lab", "title": "Python 数值验证", "content": "用数值表观察 x 趋近 0 时表达式的变化，辅助理解极限。"},
        ]


class PlannerAgent:
    def run(self, profile: Profile, assessment: Dict[str, Any], resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        rec = assessment.get("recommendation", "extra_coaching")
        cid = assessment.get("target_concept", "limit-1.1.1")
        if rec == "pass":
            steps = ["进入下一知识点", "完成 1 道迁移题", "更新知识图谱"]
            queue = ["variant", "reading", "mindmap"]
        elif rec == "extra_coaching":
            steps = ["再回答 1 个追问", "做 2 道同类题", "检查适用条件"]
            queue = ["flashcard", "exercise", "mindmap"]
        elif rec == "remediate":
            steps = ["回看结构化讲义", "完成基础题", "再进行一次诊断"]
            queue = ["lecture", "flashcard", "exercise"]
        else:
            steps = ["回到前置概念", "用直观例子重建理解", "再尝试当前概念"]
            queue = ["lecture", "animation_script", "mindmap"]
        return {
            "current_level": profile.progress_level,
            "target_concept": cid,
            "next_steps": steps,
            "resource_queue": [r for r in resources if r["type"] in queue],
            "promotion_condition": "连续 2 轮评估 score ≥ 0.8 且无 prerequisite 类错误",
            "adjustment_strategy": "根据下一轮回答动态调整追问深度、资源类型和知识树节点。"
        }


class QualityGateAgent:
    def run(self, coach: Dict[str, Any], resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        issues = []
        if len(resources) < 5:
            issues.append({"severity": "error", "description": "资源类型不足 5 类", "suggestion": "补充 flashcard、case、variant 等资源"})
        if not coach.get("message"):
            issues.append({"severity": "error", "description": "缺少追问消息", "suggestion": "生成苏格拉底式追问"})
        return {"passed": not any(i["severity"] == "error" for i in issues), "issues": issues}


def serialize_profile(p: Profile) -> Dict[str, Any]:
    d = asdict(p)
    d["blind_spots"] = [asdict(b) if not isinstance(b, dict) else b for b in p.blind_spots]
    d["knowledge_mastery_named"] = [
        {"concept_id": cid, "name": KNOWLEDGE_GRAPH[cid]["name"], "score": round(score, 2)}
        for cid, score in p.knowledge_mastery.items()
    ]
    return d

profile_builder = ProfileBuilderAgent()
diagnostician = DiagnosticianAgent()
assessor = AssessorAgent()
coach = SocraticCoachAgent()
resource_gen = ResourceGeneratorAgent()
planner = PlannerAgent()
quality_gate = QualityGateAgent()

@app.post("/api/chat")
def chat(req: ChatRequest):
    state = get_session(req.session_id)
    if req.major:
        state.profile.major = req.major
    if req.goal:
        state.profile.learning_goal = req.goal
    state.messages.append({"role": "student", "content": req.message})
    diagnosis = diagnostician.run(req.message, state.profile)
    assessment = assessor.run(req.message, diagnosis)
    profile_delta = profile_builder.run(state, req.message, {**assessment, "blind_spots": diagnosis["blind_spots"]})
    resources = resource_gen.run(state.profile, diagnosis, assessment)
    coach_out = coach.run(req.message, state.profile, assessment)
    quality = quality_gate.run(coach_out, resources)
    path = planner.run(state.profile, assessment, resources)
    state.last_assessment = assessment
    state.resource_queue = path["resource_queue"]
    state.path = path
    state.messages.append({"role": "coach", "content": coach_out["message"]})
    return {
        "session_id": state.session_id,
        "reply": coach_out["message"],
        "profile": serialize_profile(state.profile),
        "profile_delta": profile_delta,
        "diagnosis": diagnosis,
        "assessment": assessment,
        "coach": coach_out,
        "resources": resources,
        "path": path,
        "quality": quality,
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
        "last_assessment": state.last_assessment,
        "path": state.path,
        "resource_queue": state.resource_queue,
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>苏格拉底教练 · 多智能体高数辅导系统</title>
<style>
:root{--bg:#0b1020;--card:#121a33;--muted:#90a4c3;--line:#253250;--text:#eaf1ff;--brand:#67e8f9;--brand2:#a78bfa;--ok:#34d399;--warn:#fbbf24;--bad:#fb7185}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#1e3a8a55,transparent 35%),linear-gradient(135deg,#08111f,#0b1020 55%,#111827);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",Arial,sans-serif}.wrap{max-width:1280px;margin:auto;padding:28px}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:24px;align-items:center;margin-bottom:24px}.badge{display:inline-flex;gap:8px;border:1px solid #315074;background:#10233d;padding:8px 12px;border-radius:999px;color:#bde7ff;font-size:13px}h1{font-size:48px;margin:16px 0 10px}.grad{background:linear-gradient(90deg,var(--brand),var(--brand2));-webkit-background-clip:text;color:transparent}.sub{color:#bfcae0;font-size:18px;line-height:1.8}.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:20px}.card{background:linear-gradient(180deg,#141d38ee,#10182fee);border:1px solid var(--line);border-radius:24px;box-shadow:0 20px 50px #0005;padding:20px}.features{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}.feature{padding:16px;border-radius:18px;border:1px solid #2a3b5c;background:#0d172c}.feature b{display:block;margin-bottom:8px}.chat{height:430px;overflow:auto;border:1px solid var(--line);border-radius:18px;padding:16px;background:#0a1224}.msg{margin:10px 0;display:flex}.msg.student{justify-content:flex-end}.bubble{max-width:82%;padding:12px 14px;border-radius:16px;line-height:1.7}.student .bubble{background:#2563eb}.coach .bubble{background:#1f2937;border:1px solid #374151}.form{display:flex;gap:10px;margin-top:12px}input,textarea,select{background:#0b1327;border:1px solid #30405f;border-radius:14px;color:var(--text);padding:12px;outline:none}textarea{flex:1;min-height:52px;resize:vertical}button{border:0;border-radius:14px;background:linear-gradient(90deg,#06b6d4,#8b5cf6);color:white;font-weight:700;padding:0 18px;cursor:pointer}.meta{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.tab{border:1px solid #33415f;background:#0b1327;color:#cfe1ff;border-radius:999px;padding:8px 12px;cursor:pointer}.tab.active{background:#1d4ed8}.panel{display:none}.panel.active{display:block}.krow{display:grid;grid-template-columns:140px 1fr 46px;gap:10px;align-items:center;margin:10px 0;color:#dbeafe}.bar{height:10px;background:#1f2937;border-radius:999px;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,#22d3ee,#a78bfa)}.pill{display:inline-flex;margin:4px 6px 4px 0;padding:6px 10px;border-radius:999px;background:#172554;border:1px solid #315074;color:#c7d2fe;font-size:13px}.resource{border:1px solid #33415f;background:#0c162b;border-radius:16px;padding:13px;margin:10px 0}.resource h4{margin:0 0 6px}.path li{margin:10px 0}.json{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#c4d3ee;max-height:360px;overflow:auto}.footer{color:#7f91b4;text-align:center;margin-top:24px}.radar{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.dim{padding:12px;border:1px solid #33415f;border-radius:14px;background:#0b1327}.dim small{display:block;color:var(--muted);margin-bottom:6px}@media(max-width:900px){.hero,.grid{grid-template-columns:1fr}.features{grid-template-columns:1fr}.meta{grid-template-columns:1fr}h1{font-size:36px}}
</style>
</head>
<body><div class="wrap">
<section class="hero"><div><div class="badge">第十五届中国软件杯 · 科大讯飞出题 · 完整增强版</div><h1><span class="grad">苏格拉底教练</span><br/>多智能体高数辅导系统</h1><p class="sub">通过学生自然语言对话自动构建学习画像，诊断知识盲区，生成个性化资源，规划动态学习路径，并持续评估学习效果。</p><div class="features"><div class="feature"><b>画像自构建</b><span>8 维画像动态刷新</span></div><div class="feature"><b>多智能体协作</b><span>诊断、追问、资源、规划、质检闭环</span></div><div class="feature"><b>个性化推送</b><span>讲义、练习、导图、阅读、动画、代码实验</span></div></div></div><div class="card"><h3>当前系统能力</h3><span class="pill">ProfileBuilder</span><span class="pill">Diagnostician</span><span class="pill">SocraticCoach</span><span class="pill">Assessor</span><span class="pill">ResourceGenerator</span><span class="pill">Planner</span><span class="pill">QualityGate</span><p class="sub">已满足：不少于 6 个画像维度、不少于 5 类资源、路径规划、资源推送、学习效果评估。</p></div></section>
<section class="grid"><div class="card"><h2>开始学习</h2><div class="meta"><input id="major" placeholder="专业，如：计算机 / 电子 / 金融"/><input id="goal" placeholder="学习目标，如：考研 / 期末 / 补基础"/></div><div class="chat" id="chat"><div class="msg coach"><div class="bubble">你好，我会先通过追问判断你对“极限与连续”的理解。你可以直接说：我不懂等价无穷小，或尝试回答一道题。</div></div></div><div class="form"><textarea id="input" placeholder="输入你的想法，例如：我觉得 sinx 可以直接换成 x，但不太懂什么时候能用"></textarea><button onclick="send()">发送</button></div></div><div class="card"><div class="tabs"><button class="tab active" onclick="show('profile')">学习画像</button><button class="tab" onclick="show('diagnosis')">诊断评估</button><button class="tab" onclick="show('resources')">资源推送</button><button class="tab" onclick="show('path')">学习路径</button><button class="tab" onclick="show('raw')">JSON</button></div><div id="profile" class="panel active"><h3>画像维度</h3><div class="radar" id="dims"></div><h3>知识掌握</h3><div id="mastery"></div></div><div id="diagnosis" class="panel"><h3>盲区与评估</h3><div id="diag"></div></div><div id="resources" class="panel"><h3>个性化资源</h3><div id="res"></div></div><div id="path" class="panel"><h3>动态学习路径</h3><ol class="path" id="steps"></ol><div id="queue"></div></div><div id="raw" class="panel"><pre class="json" id="json"></pre></div></div></section><div class="footer">© 苏格拉底教练 · Multi-Agent High Math Tutor</div></div>
<script>
let sessionId = localStorage.getItem('hm_session_id') || null;let last=null;
function addMsg(role, text){const box=document.getElementById('chat');const div=document.createElement('div');div.className='msg '+role;div.innerHTML='<div class="bubble">'+escapeHtml(text)+'</div>';box.appendChild(div);box.scrollTop=box.scrollHeight}
function escapeHtml(s){return s.replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
async function send(){const input=document.getElementById('input');const text=input.value.trim();if(!text)return;addMsg('student',text);input.value='';const body={session_id:sessionId,message:text,major:document.getElementById('major').value,goal:document.getElementById('goal').value};const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();last=data;sessionId=data.session_id;localStorage.setItem('hm_session_id',sessionId);addMsg('coach',data.reply);render(data)}
function show(id){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));event.target.classList.add('active');document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active')}
function render(data){renderProfile(data.profile,data.profile_delta);renderDiagnosis(data);renderResources(data.resources);renderPath(data.path);document.getElementById('json').textContent=JSON.stringify(data,null,2)}
function renderProfile(p,delta){const dims=delta.dimensions||{};document.getElementById('dims').innerHTML=Object.entries(dims).map(([k,v])=>'<div class="dim"><small>'+k+'</small><b>'+v+'</b></div>').join('');document.getElementById('mastery').innerHTML=(p.knowledge_mastery_named||[]).map(x=>'<div class="krow"><span>'+x.name+'</span><div class="bar"><span style="width:'+(x.score*100)+'%"></span></div><b>'+x.score+'</b></div>').join('')}
function renderDiagnosis(d){const bs=d.diagnosis.blind_spots||[];document.getElementById('diag').innerHTML='<p><b>评分：</b>'+d.assessment.score+' / 1.0　<b>建议：</b>'+d.assessment.recommendation+'</p>'+(bs.length?bs.map(b=>'<div class="resource"><h4>'+b.concept_name+' · '+b.error_type+'</h4><p>'+b.evidence+'</p><small>根因：'+b.root_concept+'；置信度：'+b.confidence+'</small></div>').join(''):'<p>暂无明显盲区，可以进入下一层追问。</p>')+'<h4>质量把关</h4>'+((d.quality.issues||[]).length?d.quality.issues.map(i=>'<p>'+i.severity+'：'+i.description+'</p>').join(''):'<p>通过。</p>')}
function renderResources(resources){document.getElementById('res').innerHTML=resources.map(r=>'<div class="resource"><h4>'+label(r.type)+'｜'+r.title+'</h4><p>'+r.content+'</p></div>').join('')}
function renderPath(path){document.getElementById('steps').innerHTML=(path.next_steps||[]).map(s=>'<li>'+s+'</li>').join('');document.getElementById('queue').innerHTML='<h4>推荐资源队列</h4>'+(path.resource_queue||[]).map(r=>'<span class="pill">'+label(r.type)+'</span>').join('')+'<p><b>晋级条件：</b>'+path.promotion_condition+'</p>'}
function label(t){return {lecture:'讲义',exercise:'练习',mindmap:'思维导图',reading:'拓展阅读',flashcard:'速记卡',variant:'错题变式',case:'专业案例',animation_script:'动画脚本',code_lab:'代码实操'}[t]||t}
document.getElementById('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}})
</script></body></html>'''
