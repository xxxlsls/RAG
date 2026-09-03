#工厂函数
from functools import cache

from services.file_import_service import ImportFileService


@cache
def get_import_file_service() ->ImportFileService:

    return ImportFileService()