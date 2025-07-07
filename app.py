import streamlit as st
from rag_system import RAGSystem

st.set_page_config(page_title="RAG智能对话系统", layout="wide")
st.title("🤖 RAG智能对话系统")
st.markdown("支持多轮对话、文档上传、智能检索与大模型问答。")

# 初始化RAG系统
@st.cache_resource
def get_rag():
    return RAGSystem()

rag = get_rag()

# 文档上传与重建
st.sidebar.header("📄 上传文档")
uploaded_files = st.sidebar.file_uploader(
    "支持PDF、Word、TXT、CSV、HTML、Markdown", type=["pdf", "docx", "txt", "csv", "html", "md"], accept_multiple_files=True
)

if uploaded_files:
    save_dir = rag.data_path
    for file in uploaded_files:
        with open(f"{save_dir}/{file.name}", "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("文档上传成功，请点击下方按钮重建知识库。")
    if st.sidebar.button("🔄 重建知识库"):
        with st.spinner("正在重建数据库，请稍候..."):
            rag.clear_cache()
            rag._build_vectorstore()
        st.sidebar.success("✅ 向量库构建完成")

# 对话历史存储
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 展示对话历史（气泡样式）
st.header("💬 智能对话")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"].replace('\n', '<br>'), unsafe_allow_html=True)

# 输入框
question = st.chat_input("请输入您的问题...")

if question:
    # 1. 先把用户消息加入历史并立刻刷新页面
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.rerun()  # 立刻刷新页面，用户消息马上显示

# 2. 检查是否有未回复的用户消息
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
    and (
        len(st.session_state.chat_history) == 1
        or st.session_state.chat_history[-2]["role"] == "assistant"
    )
):
    with st.spinner("正在生成回答..."):
        # 构造history参数
        history = []
        msgs = st.session_state.chat_history
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                history.append([msgs[i]["content"], msgs[i+1]["content"]])
        # 发送到RAG
        answer = rag.ask(st.session_state.chat_history[-1]["content"], history=history)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()  # 立刻刷新页面，AI回复马上显示

# 展示知识库状态
with st.expander("📊 知识库状态"):
    stats = rag.get_stats()
    st.json(stats)