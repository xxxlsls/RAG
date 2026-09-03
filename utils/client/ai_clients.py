"""
  @Author:LiShuo
  @Time:2026/8/20
  @Desc:
"""


import os
import threading
from typing import Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker

from utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端：OpenAI(VLM)"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm: Optional[ChatOpenAI] = None
    _openai_llm_lock = threading.Lock()
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    @classmethod
    def get_openai(cls) -> OpenAI:
        return cls._get_or_create(
            "_openai_client", cls._openai_lock, cls._create_openai
        )

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    @classmethod
    def get_openai_llm(cls, response_format=False) -> ChatOpenAI:
        return cls._get_or_create(
            "_openai_llm", cls._openai_llm_lock,
            lambda: cls._create_openai_llm(response_format)
        )
    
    @classmethod
    def _create_openai_llm(cls, response_format=False) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("ITEM_MODEL")
    
            kwargs = dict(api_key=api_key, base_url=base_url, model=model_name, temperature=0.0)
            if response_format:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    
            client = ChatOpenAI(**kwargs)
            logger.info(f"ChatOpenAI LLM 客户端初始化成功 (model={model_name}, base_url={base_url})")
            return client
    
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI LLM 客户端创建失败: {e}")
            raise ConnectionError(f"ChatOpenAI LLM 连接失败: {e}") from e
    
    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create(
            "_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3
        )
    
    @classmethod
    def _create_bge_m3(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3")
            device = os.getenv("BGE_DEVICE", "cpu")
            use_fp16 = os.getenv("BGE_FP16", "true").lower() == "true"
    
            client = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
            )
            logger.info(f"BGE-M3 嵌入模型加载成功 (model={model_name}, device={device}, fp16={use_fp16})")
            return client
    
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-M3 模型加载失败: {e}")
            raise ConnectionError(f"BGE-M3 加载失败: {e}") from e

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create(
            "_bge_m3_rerank_client", cls._bge_m3_rerank_lock, cls._create_bge_m3_rerank_client
        )

    @classmethod
    def _create_bge_m3_rerank_client(cls) -> FlagReranker:
        try:
            model_name_or_path = cls._require_env("BGE_RERANKER_LARGE")
            device = cls._require_env("BGE_RERANKER_DEVICE")
            fp16_str = cls._require_env("BGE_RERANKER_FP16")
            fp16 = fp16_str.lower() in ("true", "1")

            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16
            )
            logger.info(f"BGE-Reranker 重排序模型加载成功 (model={model_name_or_path}, device={device}, fp16={fp16})")
            return reranker

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-Reranker 模型加载失败: {e}")
            raise ConnectionError(f"BGE-Reranker 加载失败: {e}") from e
