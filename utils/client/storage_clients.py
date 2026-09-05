"""
  @Author:LiShuo
  @Time:2026/8/20
  @Desc:
"""
import json
# knowledge/utils/client/storage_clients.py

import threading
from typing import Optional
import logging

from minio import Minio
from pymongo import MongoClient
from pymongo.database import Database
from pymilvus import MilvusClient
from dotenv import load_dotenv
from utils.client.base import BaseClientManager

logger = logging.getLogger(__name__)
load_dotenv()


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus、MongoDB"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    _mongo_db: Optional[Database] = None
    _mongo_lock = threading.Lock()

    @classmethod
    def get_mongo_db(cls) -> Database:
        return cls._get_or_create(
            "_mongo_db", cls._mongo_lock, cls._create_mongo_db
        )

    @classmethod
    def _create_mongo_db(cls) -> Database:
        try:
            mongo_url = cls._require_env("MONGO_URL")
            db_name = cls._require_env("MONGO_DB_NAME")
            client = MongoClient(mongo_url)
            db = client[db_name]
            logger.info(f"MongoDB 客户端初始化成功 (url={mongo_url}, db={db_name})")
            return db

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MongoDB 客户端创建失败: {e}")
            raise ConnectionError(f"MongoDB 连接失败: {e}") from e

    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create(
            "_milvus_client", cls._milvus_lock, cls._create_milvus
        )

    @classmethod
    def _create_milvus(cls) -> MilvusClient:
        try:
            milvus_url = cls._require_env("MILVUS_URL")
            client = MilvusClient(uri=milvus_url)
            logger.info(f"Milvus 客户端初始化成功 (uri={milvus_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e

    @classmethod
    def get_minio(cls) -> Minio:
        return cls._get_or_create(
            "_minio_client", cls._minio_lock, cls._create_minio
        )

    @classmethod
    def _create_minio(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(
                endpoint, access_key=access_key,
                secret_key=secret_key, secure=False
            )

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise  # 配置缺失，直接往上抛
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e
    @staticmethod
    def _ensure_public_read(client: Minio, bucket_name: str) -> None:
        """设置桶的匿名只读策略。

        前端 <img src> 是不带凭证的匿名请求，而 MinIO 桶默认 private，
        会返回 403 AccessDenied，表现为图片裂开。
        只放开 s3:GetObject，不放开 s3:ListBucket，避免对象列表被枚举。
        """
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }],
        }
        try:
            client.set_bucket_policy(bucket_name, json.dumps(policy))
            logger.info(f"MinIO bucket '{bucket_name}' 匿名只读策略已生效")
        except Exception as e:
            logger.warning(f"设置匿名只读策略失败，前端图片可能返回 403: {e}")