import os



KNOWLEDGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地临时数据存储的基础目录
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")
# 首页相关资源所在的目录
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")

def get_local_base_dir() -> str:
    """获取本地临时数据存储的基础目录路径"""
    return LOCAL_BASE_DIR

def get_front_page_dir() -> str:
    """获取首页相关资源所在的目录路径"""
    return FRONT_PAGE_DIR
