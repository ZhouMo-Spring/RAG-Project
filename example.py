#通过命令行交互
from rag_system import RAGSystem


def main():
    # 创建RAG系统实例（使用全局知识库）
    print("🚀 初始化RAG系统...")
    rag = RAGSystem()  # 不传user_id则使用全局知识库
    
    # 显示系统信息
    stats = rag.get_stats()
    print(f"\n📊 系统信息:")
    print(f"  - 用户ID: {stats['user_id']}")
    print(f"  - 数据路径: {stats['data_path']}")
    print(f"  - 向量库状态: {'已构建' if stats['vectorstore_exists'] else '未构建'}")
    print(f"  - 文档数量: {stats['document_count']}")
    print(f"  - Embedding模型: {stats['embedding_model']}")
    
    # 示例问题
    questions = [
        "高血压产生的原因是什么",
        "如何治疗",
        "饮食上有什么需要注意的吗"
    ]
    
    print(f"\n=== 开始问答测试 ===")
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)
        
        # 直接调用ask方法，无需复杂的调用链
        answer = rag.ask(question)
        
        print(f"回答: {answer}")
        print("-" * 50)


def test_user_specific():
    """测试用户特定的RAG系统"""
    print("\n=== 用户特定RAG系统测试 ===")
    
    # 创建用户特定的RAG系统
    user_rag = RAGSystem(user_id="test_user")
    
    # 显示用户系统信息
    stats = user_rag.get_stats()
    print(f"用户系统信息: {stats}")
    
    # 测试问答
    question = "请介绍一下智能体的概念"
    answer = user_rag.ask(question)
    print(f"用户特定回答: {answer}")


if __name__ == "__main__":
    main()
    
    # 可选：测试用户特定系统
    # test_user_specific() 