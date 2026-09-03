import os
import subprocess
from pathlib import Path

from processor.import_process.exceptions import FileProcessingError
from processor.import_process.state import ImportGraphState

from processor.import_process.base import BaseNode


class PdfToMdNode(BaseNode):
    name:str = "pdf_to_md_node"

    def process(self, state:ImportGraphState) -> dict:

        # 从state["pdf_path"]拿到PDF路径
        p = Path(state["pdf_path"])

        # 确定输出目录（用state["file_dir"]或PDF所在目录
        pdf_output_path = state["file_dir"]

        # 设置环境变量(让mineru)用离线模型
        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = "modelscope"
        env["MODELSCOPE_OFFLINE"] = "1"
        env["PATH"] = r"D:\py_workspace\conda_envs\knowledge\Scripts;" + env.get("PATH", "")

        # 用subprocess.run()调用mineru命令
        cmd = [
            "mineru",
            '-p', str(p),  # pdf路径
            "-o", pdf_output_path,  # 输出目录
            "--backend", "pipeline"  # 使用本地模型
        ]

        result = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,  # ← 代替 capture_output
            stderr=subprocess.STDOUT,  # ← 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1  # 行缓冲，实时输出
        )
        # 逐行打印 MinerU 的实时输出
        for line in result.stdout:
            print(line.rstrip())
        # 等待进程结束，获取退出码
        return_code = result.wait()

        # 检查执行结果
        if result.returncode != 0:
            from processor.import_process.exceptions import PdfConversionError
            raise PdfConversionError(
                f'MinerU 转换失败: {return_code}',
                self.name
            )

        # 构造生成的md文件路径
        pdf_stem = p.stem
        md_file = Path(pdf_output_path) /pdf_stem /'auto'/ f'{pdf_stem}.md'

        # 文件存在性检查
        if not md_file.exists():
            raise FileProcessingError(
                f'MinerU未生成md文件: {md_file}',
                self.name
            )

        # 读取md内容
        md_content = md_file.read_text(encoding='utf-8')

        return {
            'md_content': md_content,
            'md_path': str(md_file),
        }