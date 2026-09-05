#工厂函数
from functools import lru_cache

from services.file_import_service import ImportFileService
from services.query_service import QueryService


@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService()

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()