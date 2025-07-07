#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM客户端类
支持智谱AI GLM-4模型
"""

import os
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def chat_with_ai_stream(self, prompt: str) -> str:
        """与AI聊天的方法"""
        pass

class GLM4Client(BaseLLMClient):
    """智谱AI GLM-4客户端"""
    
    def __init__(self):
        # 从环境变量获取配置
        self.api_key = os.getenv('LLM_API_KEY')
        self.base_url = os.getenv('LLM_BASE_URL')
        self.model_name = os.getenv('MODEL_NAME', 'GLM-4-Flash-250414')
        
        if not self.api_key or not self.base_url:
            raise ValueError("请在.env文件中配置LLM_API_KEY和LLM_BASE_URL")
        
        # 创建ChatOpenAI实例
        self.llm = ChatOpenAI(
            temperature=0.7,
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url
        )
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    # 通用system prompt
                    "你是一个专业的AI助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。"
                    # 医疗领域system prompt
                    # "你是一名医学AI助手，请严格根据提供的医学文档内容回答问题，所有内容仅供医学参考。如无相关信息，请说明无法从提供的信息中找到答案。"
                ),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        
        # 使用新的RunnableSequence语法
        self.chain = self.prompt | self.llm
    
    def chat_with_ai_stream(self, prompt: str) -> str:
        """与GLM-4聊天"""
        try:
            response = self.chain.invoke({"question": prompt})
            return response.content
        except Exception as e:
            return f"GLM-4调用失败: {str(e)}"


def create_llm_client() -> BaseLLMClient:
    """创建LLM客户端工厂函数"""
    try:
        # 尝试创建GLM-4客户端
        return GLM4Client()
    except Exception as e:
        print(f"创建GLM-4客户端失败: {e}")