# 苏格拉底教练 · 多智能体信号与系统辅导系统

这是在原型基础上补全后的成品版本，覆盖：

- 对话式学习画像自动构建与动态刷新
- 多智能体协作：画像、诊断、追问、评估、资源生成、路径规划、质量把关
- 不少于 6 个画像维度
- 不少于 5 类个性化学习资源
- 个性化学习路径规划与资源推送
- 学习效果评估与下一步策略调整

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

打开：`http://127.0.0.1:8000`

## Railway 部署

直接上传本目录，Railway 会使用 `Procfile` 启动。

## 主要接口

- `GET /` 前端页面
- `POST /api/chat` 学生对话输入，返回多智能体结果
- `GET /api/profile/{session_id}` 获取画像
- `GET /api/export/{session_id}` 导出完整学习报告 JSON
