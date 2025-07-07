# 📚 RAG 智能问答系统

本项目是一个基于检索增强生成（RAG, Retrieval-Augmented Generation）的智能问答系统，支持多格式文档上传、向量化、智能检索与大模型问答。  
前后端一体，前端采用 Streamlit，后端支持多轮对话、文档片段可溯源、向量库缓存等特性。

---

## 🚀 主要特性

- **多格式文档支持**：PDF、Word、TXT、CSV、HTML、Markdown
- **智能向量化检索**：BGE中文embedding + FAISS向量数据库
- **大模型问答**：支持智谱AI GLM-4（可扩展OpenAI等）
- **多轮对话**：前端对话框体验，历史上下文自动带入
- **文档片段可溯源**：回答后自动显示涉及的文档及页码
- **向量库缓存**：自动缓存，文档无变化时秒级启动
- **Streamlit前端**：支持文件上传、知识库重建、对话气泡UI

---

## 🖥️ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制`.env_example`，在项目根目录创建 `.env` 文件，填写你的API 秘钥和选择的模型：

```env
LLM_API_KEY=你的智谱API密钥
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODEL_NAME=GLM-4-Flash-250414
```

### 3. 配置知识库路径和Embedding模型

编辑 `config/config-web.yaml`，如：

```yaml
Knowledge-base-path: ./KnowledgeBase

model:
  embedding:
    model-name: BAAI/bge-base-zh-v1.5
    device: cpu
```

### 4. 启动前端

```bash
streamlit run app.py
```

---

## 📝 使用说明

### 文档上传与知识库重建

- 在侧边栏上传PDF、Word、TXT等文档
- 上传后点击“重建知识库”，系统会自动分割、向量化并缓存
- 重建时会显示“正在重建数据库，请稍候...”，完成后显示“✅ 向量库构建完成”

### 智能对话

- 在主界面输入问题，点击发送
- 支持多轮对话，历史自动带入
- 回答下方会显示本次检索到的文档片段（来源、页码）

### 知识库状态

- 展开“知识库状态”可查看当前文档数量、embedding模型等信息

---

## 📦 目录结构

```
RAG_Project/
├── app.py                # Streamlit前端主程序
├── rag_system.py         # RAG核心后端（检索、向量化、缓存、对话）
├── llm_client.py         # LLM API客户端（支持智谱GLM-4）
├── config/
│   └── config-web.yaml   # 配置文件
├── KnowledgeBase/        # 知识库文档目录
├── vector_cache/         # 向量库缓存目录（自动生成）
├── requirements.txt      # 依赖包
└── README.md             # 项目说明
```

---

## ⚡ 依赖说明

详见 `requirements.txt`，主要包括：

- streamlit
- langchain-community / langchain-core / langchain-huggingface / langchain-openai
- faiss-cpu
- huggingface-hub / transformers / torch / sentence-transformers
- unstructured / python-docx / markdown
- openai / scikit-learn / numpy / PyYAML / python-dotenv

---

## 🧠 进阶功能

- **多用户隔离**：可扩展为多用户独立知识库
- **本地/私有模型**：可扩展为本地LLM
- **对话导出**：可扩展为导出历史对话
- **UI美化**：可自定义Streamlit主题和气泡样式

---

## 🐛 常见问题

- **向量库构建慢**：首次构建需分割和向量化，后续无文档变化则秒级启动
- **LLM调用失败**：请检查API密钥和网络
- **文档未被检索到**：请确认文档格式和内容

---

## 📄 许可证

MIT License

---

如需进一步定制、扩展或遇到问题，欢迎随时提问！