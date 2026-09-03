import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，避免被 site-packages 中同名 processor 包遮蔽
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import FileProcessingError, ValidationError
from processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name: str = "entry_node"

    # 定义一个函数 entry_node(state: ImportGraphState) -> dict
    def process(self, state:ImportGraphState) ->ImportGraphState|dict:
        # 从 state["import_file_path"] 拿到文件路径       import_file_path: str  # 导入文件路径
        p = Path(state["import_file_path"])
        if  not state.get('file_dir'):
            raise ValidationError("数据校验失败:import_file_path为空", self.name)
        # 判断文件后缀：
        # .pdf → 返回 {"is_pdf_read_enabled": True, "is_md_read_enabled": False, "pdf_path": ..., "file_title": ..., "file_dir": ...}
        if p.suffix.lower() == ".pdf":
            return {
                "is_pdf_read_enabled": True,
                "is_md_read_enabled": False,
                "pdf_path": str(p),
                "file_title":p.stem,
                "file_dir":str(p.parent),
            }
        elif p.suffix.lower() == ".md":
            return {
                "is_pdf_read_enabled": False,
                "is_md_read_enabled": True,
                "md_path": str(p),
                "file_title": p.stem,
                "file_dir": str(p.parent),
            }

        # 其他 → 抛异常
        else:
            raise FileProcessingError("文件处理错误：文件不存在、格式错误、读写失败",self.name)

if __name__ == '__main__':
    setup_logging()

    input_state = {
        "import_file_path": r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用.pdf",  # 导入文件路径（原始输入）
        "file_dir": r"E:\temp_dir",  # 导入(出)文件目录
    }

    entry = EntryNode()
    # 自动调用父类__call__()
    node_state = entry(input_state)

    print(json.dumps(node_state, ensure_ascii=False, indent=4))