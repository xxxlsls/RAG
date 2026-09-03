"""
  @Author:LiShuo
  @Time:2026/8/18
  @Desc:
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_milvus():
    print("测试 Milvus 连接...")
    try:
        from pymilvus import MilvusClient
        uri = os.getenv("MILVUS_URL", "http://192.168.10.170:19530")
        client = MilvusClient(uri=uri)
        version = client.get_server_version()
        print(f"  ✓ Milvus 连接成功，版本: {version}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ Milvus 连接失败: {e}")
        return False

def test_mongodb():
    print("测试 MongoDB 连接...")
    try:
        from pymongo import MongoClient
        url = os.getenv("MONGO_URL", "mongodb://192.168.10.170:27017")
        client = MongoClient(url, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db_names = client.list_database_names()
        print(f"  ✓ MongoDB 连接成功，数据库列表: {db_names}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ MongoDB 连接失败: {e}")
        return False

def test_minio():
    print("测试 MinIO 连接...")
    try:
        from minio import Minio
        endpoint = os.getenv("MINIO_ENDPOINT", "192.168.10.170:9000")
        client = Minio(endpoint,
                       access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                       secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                       secure=False)
        buckets = client.list_buckets()
        print(f"  ✓ MinIO 连接成功，存储桶: {[b.name for b in buckets]}")
        return True
    except Exception as e:
        print(f"  ✗ MinIO 连接失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("掌柜智库 - 服务连接测试")
    print("=" * 50)
    results = {"Milvus": test_milvus(), "MongoDB": test_mongodb(), "MinIO": test_minio()}
    print("\n" + "=" * 50)
    for svc, passed in results.items():
        print(f"  {svc}: {'✓ 通过' if passed else '✗ 失败'}")
    print("=" * 50)