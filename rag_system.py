"""
整合的RAG系统
将检索、向量化、LLM调用等功能整合在一个类中，简化调用关系
支持向量库持久化存储
"""

import os
import pickle
import hashlib
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader,
    TextLoader, CSVLoader, UnstructuredHTMLLoader, MHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.retriever import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import Config
from llm_client import create_llm_client, BaseLLMClient


class RAGSystem:
    """整合的RAG系统类"""
    
    def __init__(self, user_id: Optional[str] = None):
        """
        初始化RAG系统
        
        Args:
            user_id: 用户ID，如果为None则使用全局知识库
        """
        self.user_id = user_id
        self.config = Config.get_instance()
        
        # 初始化embedding模型
        self._init_embedding()
        
        # 初始化LLM客户端
        self._init_llm_client()
        
        # 初始化路径
        self._init_paths()
        
        # 存储向量库
        self._vectorstores: Dict[str, FAISS] = {}
        self._retrievers: Dict[str, BaseRetriever] = {}
        
        # 尝试加载缓存的向量库，如果失败则重新构建
        if not self._load_cached_vectorstore():
            self._build_vectorstore()
    
    def _init_embedding(self):
        """初始化embedding模型"""
        embedding_model_name = self.config.get_with_nested_params(
            "model", "embedding", "model-name"
        )
        device = self.config.get_with_nested_params(
            "model", "embedding", "device"
        )
        
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ 初始化embedding模型: {embedding_model_name}")
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        self.llm_client = create_llm_client()
        print("✅ 初始化LLM客户端")
    
    def _init_paths(self):
        """初始化路径"""
        if self.user_id:
            # 用户特定路径
            self.data_path = os.path.join("user_data", self.user_id)
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
                print(f"📁 创建用户文件夹: {self.data_path}")
        else:
            # 全局路径
            self.data_path = self.config.get_with_nested_params("Knowledge-base-path")
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
                print(f"📁 创建知识库文件夹: {self.data_path}")
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        print(f"📚 从 {self.data_path} 加载文档...")
        
        # 统计信息
        file_count = 0
        page_count = 0
        all_docs = []
        
        # 加载各种格式的文档
        loaders = [
            DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True),
            DirectoryLoader(self.data_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, silent_errors=True),
            DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True, loader_kwargs={"autodetect_encoding": True}),
            DirectoryLoader(self.data_path, glob="**/*.csv", loader_cls=CSVLoader, silent_errors=True, loader_kwargs={"autodetect_encoding": True}),
            DirectoryLoader(self.data_path, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, silent_errors=True),
            DirectoryLoader(self.data_path, glob="**/*.mhtml", loader_cls=MHTMLLoader, silent_errors=True),
            DirectoryLoader(self.data_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, silent_errors=True),
        ]
        
        for loader in loaders:
            try:
                docs = loader.load()
                if docs:
                    # 统计文件数量（通过文件名去重）
                    unique_files = set()
                    for doc in docs:
                        # 正确访问metadata
                        if 'source' in doc.metadata:
                            unique_files.add(doc.metadata['source'])
                        elif 'file_path' in doc.metadata:
                            unique_files.add(doc.metadata['file_path'])
                        elif 'file_name' in doc.metadata:
                            unique_files.add(doc.metadata['file_name'])
                    
                    file_count += len(unique_files)
                    page_count += len(docs)
                    all_docs.extend(docs)
                        
            except Exception as e:
                print(f"  - 加载文档时出错: {e}")
        
        print(f"📊 总共加载了 {file_count} 个文件，{page_count} 个页面/片段")
        return all_docs
    
    def _build_vectorstore(self):
        """构建向量库"""
        print("🔨 构建向量库...")
        
        # 加载文档
        docs = self._load_documents()
        
        if not docs:
            print("⚠️  没有找到文档，向量库构建失败")
            return
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)
        print(f"✂️  文档分割为 {len(splits)} 个片段")
        
        # 创建向量库
        vectorstore = FAISS.from_documents(documents=splits, embedding=self.embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # 存储向量库和检索器
        key = self.user_id or "global"
        self._vectorstores[key] = vectorstore
        self._retrievers[key] = retriever
        
        # 保存到缓存
        self._save_vectorstore_cache(vectorstore, splits)
        
        print(f"✅ 向量库构建完成 (key: {key})")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        key = self.user_id or "global"
        retriever = self._retrievers.get(key)
        
        if not retriever:
            print("❌ 向量库未构建，无法检索")
            return []
        
        try:
            docs = retriever.invoke(query)
            print(f"🔍 检索到 {len(docs)} 个相关文档片段")
            return docs
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []
    
    def format_documents(self, docs: List[Document]):
        """
        返回两个内容：
        - formatted_docs: 用于prompt的完整内容（含正文）
        - doc_infos: 仅包含doc_info（不含正文）
        """
        if not docs:
            return "没有找到相关的文档信息", []

        formatted_docs = []
        doc_infos = []
        for i, doc in enumerate(docs, 1):
            # 获取文档来源信息
            source = "未知来源"
            page = "未知页码"
            
            # 正确访问metadata
            if 'source' in doc.metadata:
                source = doc.metadata['source']
            elif 'file_path' in doc.metadata:
                source = doc.metadata['file_path']
            elif 'file_name' in doc.metadata:
                source = doc.metadata['file_name']
            
            if 'page' in doc.metadata:
                # 页码从1开始
                page_num = doc.metadata['page']
                if isinstance(page_num, int):
                    page = str(page_num + 1)  # 从0开始转换为从1开始
                else:
                    page = str(page_num)
            elif 'page_number' in doc.metadata:
                # 页码从1开始
                page_num = doc.metadata['page_number']
                if isinstance(page_num, int):
                    page = str(page_num + 1)  # 从0开始转换为从1开始
                else:
                    page = str(page_num)
            elif 'row' in doc.metadata:
                # 行数从1开始
                page_num = doc.metadata['row']
                if isinstance(page_num, int):
                    page = str(page_num + 1)  # 从0开始转换为从1开始
                else:
                    page = str(page_num)        
            
            # 提取文件名
            if source != "未知来源":
                filename = os.path.basename(source)
            else:
                filename = "未知文件"
            
            # 格式化文档片段
            doc_info = f"【文档片段 {i}】来源: {filename}, 页码/行数: {page}"
            doc_content = doc.page_content.strip()
            
            formatted_docs.append(f"{doc_info}\n{doc_content}")
            doc_infos.append(doc_info)
        
        return "\n\n-------------分割线--------------\n\n".join(formatted_docs), doc_infos
    
    def ask(self, question: str, history: Optional[list] = None) -> str:
        """
        支持多轮对话历史的问答接口
        history: List[List]，如 [["用户问题1", "助手回答1"], ["用户问题2", "助手回答2"]]
        """
        print(f"\n🔍 检索问题: {question}")

        # 检索相关文档
        docs = self.retrieve_documents(question)
        context, doc_infos = self.format_documents(docs)

        # 构建历史对话字符串
        history_str = ""
        if history:
            for i, (q, a) in enumerate(history, 1):
                history_str += f"用户: {q}\n助手: {a}\n"

        # 构建提示词
        prompt = f"""
请根据以下检索到的文档信息回答用户的新问题。

【历史对话】（如有）：
{history_str}

【检索到的文档信息】：
{context}

【新问题】：
{question}

【作答要求】：
1. 基于检索到的文档信息进行回答
2. 在回答中引用具体的文档片段编号（如"根据文档片段1"）
3. 如果信息来自多个片段，请分别引用
4. 如果文档信息不足以回答问题，请明确说明

【请基于上述文档信息作答】："""
        
#         prompt = f"""
#         你是一名专业的医学AI助手，请严格基于下方检索到的医学文档信息，科学、严谨地回答用户的新问题。

# 【历史对话】（如有）：
# {history_str}

# 【检索到的医学文档信息】：
# {context}

# 【新问题】：
# {question}

# 【作答要求】：
# 1. 仅基于检索到的医学文档内容进行回答，不要凭空编造。
# 2. 回答中请明确引用具体的文档片段编号（如“根据文档片段1”）。
# 3. 如答案涉及多个片段，请分别引用。
# 4. 如文档信息不足以回答，请直接说明“无法从提供的信息中找到答案”。
# 5. 不得提供具体诊断或治疗建议，所有内容仅供医学参考。
# 6. 回答应简明、专业、客观，避免主观臆断和夸大其词。

# 【请基于上述医学文档信息作答】："""


        print(f"🤖 调用LLM生成回答...")

        try:
            response = self.llm_client.chat_with_ai_stream(prompt)
            doc_info_str = "\n".join(doc_infos)
            final_response = response + "\n\n本次检索到的文档片段：\n" + doc_info_str
            return final_response
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            return f"抱歉，生成回答时出现错误: {e}"
        
        

    def add_documents(self, file_paths: List[str]):
        """添加新文档到向量库"""
        print("📝 添加新文档...")
        
        # 这里可以实现增量添加文档的逻辑
        # 为了简化，重新构建整个向量库
        self._build_vectorstore()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        key = self.user_id or "global"
        vectorstore = self._vectorstores.get(key)
        
        stats = {
            "user_id": self.user_id,
            "data_path": self.data_path,
            "vectorstore_exists": vectorstore is not None,
            "document_count": len(vectorstore.docstore._dict) if vectorstore else 0,
            "embedding_model": self.embedding.model_name,
        }
        
        return stats

    def _get_cache_path(self) -> str:
        """获取缓存路径"""
        key = self.user_id or "global"
        cache_dir = os.path.join("vector_cache", key)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_vectorstore_cache_path(self) -> str:
        """获取向量库缓存文件路径"""
        cache_dir = self._get_cache_path()
        return os.path.join(cache_dir, "vectorstore.pkl")
    
    def _get_documents_cache_path(self) -> str:
        """获取文档缓存文件路径"""
        cache_dir = self._get_cache_path()
        return os.path.join(cache_dir, "documents.pkl")
    
    def _get_data_hash(self) -> str:
        """计算数据目录的哈希值，用于检测文件变化"""
        if not os.path.exists(self.data_path):
            return ""
        
        hash_md5 = hashlib.md5()
        for root, dirs, files in os.walk(self.data_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        hash_md5.update(f.read())
                except:
                    pass
        
        return hash_md5.hexdigest()
    
    def _get_cache_hash_path(self) -> str:
        """获取缓存哈希文件路径"""
        cache_dir = self._get_cache_path()
        return os.path.join(cache_dir, "data_hash.txt")
    
    def _load_cached_vectorstore(self) -> bool:
        """尝试加载缓存的向量库"""
        vectorstore_path = self._get_vectorstore_cache_path()
        documents_path = self._get_documents_cache_path()
        hash_path = self._get_cache_hash_path()
        
        # 检查缓存文件是否存在
        if not (os.path.exists(vectorstore_path) and os.path.exists(documents_path) and os.path.exists(hash_path)):
            print("📋 缓存文件不存在，需要重新构建向量库")
            return False
        
        # 检查数据是否发生变化
        current_hash = self._get_data_hash()
        try:
            with open(hash_path, 'r') as f:
                cached_hash = f.read().strip()
            
            if current_hash != cached_hash:
                print("📋 数据发生变化，需要重新构建向量库")
                return False
        except:
            print("📋 无法读取缓存哈希，需要重新构建向量库")
            return False
        
        # 加载缓存的向量库
        try:
            print("📋 加载缓存的向量库...")
            
            with open(vectorstore_path, 'rb') as f:
                vectorstore = pickle.load(f)
            
            with open(documents_path, 'rb') as f:
                documents = pickle.load(f)
            
            # 重新创建检索器
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            
            # 存储到内存
            key = self.user_id or "global"
            self._vectorstores[key] = vectorstore
            self._retrievers[key] = retriever
            
            print(f"✅ 成功加载缓存的向量库，包含 {len(documents)} 个文档片段")
            return True
            
        except Exception as e:
            print(f"❌ 加载缓存失败: {e}")
            return False
    
    def _save_vectorstore_cache(self, vectorstore: FAISS, documents: List[Document]):
        """保存向量库到缓存"""
        try:
            print("💾 保存向量库到缓存...")
            
            # 保存向量库
            vectorstore_path = self._get_vectorstore_cache_path()
            with open(vectorstore_path, 'wb') as f:
                pickle.dump(vectorstore, f)
            
            # 保存文档信息
            documents_path = self._get_documents_cache_path()
            with open(documents_path, 'wb') as f:
                pickle.dump(documents, f)
            
            # 保存数据哈希
            hash_path = self._get_cache_hash_path()
            current_hash = self._get_data_hash()
            with open(hash_path, 'w') as f:
                f.write(current_hash)
            
            print("✅ 向量库缓存保存成功")
            
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
    
    def clear_cache(self):
        """清除缓存"""
        cache_dir = self._get_cache_path()
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"🗑️  已清除缓存: {cache_dir}")
        
        # 清除内存中的向量库
        key = self.user_id or "global"
        if key in self._vectorstores:
            del self._vectorstores[key]
        if key in self._retrievers:
            del self._retrievers[key]


# 全局实例
_global_rag_system: Optional[RAGSystem] = None


def get_rag_system(user_id: Optional[str] = None) -> RAGSystem:
    """获取RAG系统实例"""
    global _global_rag_system
    
    if user_id is None:
        if _global_rag_system is None or _global_rag_system.user_id is not None:
            _global_rag_system = RAGSystem()
        return _global_rag_system
    else:
        # 为每个用户创建独立的实例
        return RAGSystem(user_id) 