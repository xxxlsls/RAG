import base64
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Deque, Set

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, FileProcessingError
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


# ──────────────────── 数据模型（多个类共享 → 放模块级）────────────────────

@dataclass
class ImageContext:
    """图片在md中的上下文信息"""
    heading: str    # 最近的章节标题
    up_text: str    # 图片上方的正文内容
    down_text: str  # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名（替换MD链接时用来匹配）
    path: str                # 图片完整路径（打开文件读字节时用）
    context: ImageContext    # 在 MD 中的上下文


# ──────────────────── 1. 文件读写 & 备份 ────────────────────

class MdFileHandler:
    """负责 MD 内容读取、路径校验、图片目录构建、处理后备份。"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def read_md(self, state: dict):
        """读取 MD 内容，返回 (md_content, md_path_obj, image_dir)"""
        self.logger.info("【step_1】读取MD内容及构建图片目录")

        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(
                node_name="md_img_node", field_name="md_path", expected_type=str,
            )

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(f"md文件路径无效: {md_path}", node_name="md_img_node")

        # 兼容：state 里已有 md_content 就优先用，没有才读磁盘
        md_content = state.get("md_content")
        if not md_content:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()

        image_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, image_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """把处理后的内容另存为 原名_new.md，方便对照调试"""
        self.logger.info("【step_5】备份新文件")
        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(new_md_content)
        self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        return str(new_file_path)


# ──────────────────── 2. 图片扫描 & 上下文提取 ────────────────────

class ImageScanner:
    """扫描图片目录，提取每张图片在 MD 中的上下文信息"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def scan_img_dir(self, image_dir, md_content,
                     image_extensions: Set[str], context_length: int) -> List[ImageInfo]:
        """遍历图片目录，返回 List[ImageInfo]"""
        self.logger.info(f"【step_2】扫描图片目录 {image_dir}")
        image_list: List[ImageInfo] = []

        for image_path in Path(image_dir).iterdir():
            # ① 不是文件的跳过（目录、子文件夹）
            if not image_path.is_file():
                continue
            # ② 后缀不在 image_extensions 里的跳过（比如 .txt）
            if image_path.suffix.lower() not in image_extensions:
                continue
            # ③ 找上下文，返回 None 说明 MD 里没引用这张图
            context = self._find_context(md_content, image_path.name, context_length)
            if context is None:
                self.logger.warning(f"MD 里没引用这张图 {image_path.name}")
                continue
            # ④ 组装 ImageInfo 加入列表
            image_list.append(ImageInfo(
                name=image_path.name, path=str(image_path), context=context
            ))

        self.logger.info(f"找到 {len(image_list)} 张有效图片")
        return image_list

    def _find_context(self, md_content: str, img_name: str, context_length: int = 200):
        """提取图片在 MD 中第一次出现位置的上下文，找不到返回 None。"""
        # MD 里引用可能带路径前缀（如 images/xxx.jpg），所以文件名前允许任意非右括号字符
        pattern = r'!\[.*?]\([^)]*?' + re.escape(img_name) + r'\)'
        match = re.search(pattern, md_content)
        if match is None:
            return None

        # 向上：图片前面最近的标题（图片所属章节）
        headings = re.findall(r'^#{1,6}\s+.+', md_content[:match.start()], re.MULTILINE)

        return ImageContext(
            heading=headings[-1].strip() if headings else "",
            # 上文用 match.start() 收尾，下文用 match.end() 开头，
            # 这样 ![](xxx.jpg) 图片语法本身不会被切进上下文
            up_text=md_content[max(0, match.start() - context_length):match.start()],
            down_text=md_content[match.end():match.end() + context_length],
        )


# ──────────────────── 3. VLM 摘要生成（限流 + 两层降级）────────────────────

class VLMSummarizer:
    """通过视觉语言模型为每张图片生成中文标题/摘要。"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def summarize_all(self, document_title: str, image_list: List[ImageInfo],
                      vl_model: str, requests_per_minute: int) -> Dict[str, str]:
        """为所有图片生成摘要，返回 {图片名: 摘要}"""
        self.logger.info("【step_3】提取图片摘要")
        summaries: Dict[str, str] = {}
        request_timestamps: Deque[float] = deque()  # 滑动窗口：60秒内的请求时间戳

        # 连接级降级：VLM 整体不可用 → 全部用默认描述，不崩溃
        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(f"VLM 不可用，跳过图片摘要生成: {e}")
            for img in image_list:
                summaries[img.name] = "图片描述"
            return summaries

        for img in image_list:
            self._enforce_rate_limit(request_timestamps, requests_per_minute)
            summaries[img.name] = self._summarize_one(client, vl_model, document_title, img)

        self.logger.info(f"生成 {len(summaries)} 张图片摘要")
        return summaries

    def _summarize_one(self, client, vl_model: str, document_title: str, img: ImageInfo) -> str:
        """生成单张图片摘要；失败时个体级降级，不影响其他图片。"""
        # 把上下文拼成背景信息，帮助 VLM 理解图片
        parts = [p for p in (img.context.heading, img.context.up_text, img.context.down_text) if p]
        final_context = "\n".join(parts) if parts else "暂无可用上下文"

        try:
            with open(img.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.path}: {e}")
            return "图片描述"

    def _enforce_rate_limit(self, timestamps: Deque[float], max_requests: int, window: int = 60):
        """滑动窗口限流：窗口满了就等到最早的请求滑出窗口再放行。"""
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_dur:.2f} 秒...")
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)


# ──────────────────── 4. MinIO 上传 & MD 内容替换 ────────────────────

class ImageUploader:
    """将本地图片上传至 MinIO，并在 MD 内容中替换为远程 URL + 摘要。"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def upload_and_replace(self, document_name: str, md_content: str,
                           images_summaries: Dict[str, str], image_list: List[ImageInfo],
                           minio_bucket: str, minio_base_url: str) -> str:
        self.logger.info("【step_4】上传图片到MinIO并更新MD")
        remote_urls = self._upload_all(document_name, image_list, minio_bucket, minio_base_url)
        return self._replace_in_md(md_content, images_summaries, remote_urls)

    def _upload_all(self, document_name: str, image_list: List[ImageInfo],
                    minio_bucket: str, minio_base_url: str) -> Dict[str, str]:
        """上传全部图片，返回 {图片名: 远程URL}"""
        remote_urls: Dict[str, str] = {}

        # 连接级降级：MinIO 不可用 → 全部保留本地路径
        try:
            minio_client = StorageClients.get_minio()
        except Exception as e:
            self.logger.warning(f"MinIO 不可用，所有图片保留本地路径: {e}")
            for img in image_list:
                remote_urls[img.name] = img.path
            return remote_urls

        for img in image_list:
            object_name = f"{document_name}/{img.name}"
            # 个体级降级：单张上传失败保留本地路径，不影响其他
            try:
                minio_client.fput_object(minio_bucket, object_name, img.path)
                remote_urls[img.name] = f"{minio_base_url}/{minio_bucket}/{object_name}"
                self.logger.info(f"{img.name} 上传成功")
            except Exception:
                self.logger.warning(f"{img.name} 上传失败，保留本地路径")
                remote_urls[img.name] = img.path

        return remote_urls

    @staticmethod
    def _replace_in_md(md_content: str, summaries: Dict[str, str],
                       remote_urls: Dict[str, str]) -> str:
        """替换 MD 中的图片引用为 远程URL+摘要；不在处理清单里的保持原样。"""
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            file_name_in_md = Path(match.group(2).strip()).name
            if file_name_in_md in summaries:
                return f"![{summaries[file_name_in_md]}]({remote_urls[file_name_in_md]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


# ──────────────────── 主节点（瘦编排层：只负责串联流程）────────────────────

class MarkDownImageNode(BaseNode):
    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.file_handler = MdFileHandler(self.logger)
        self.scanner = ImageScanner(self.logger)
        self.summarizer = VLMSummarizer(self.logger)
        self.uploader = ImageUploader(self.logger)

    def process(self, state) -> dict:
        config = self.config

        # 1. 读取文件
        md_content, md_path_obj, image_dir = self.file_handler.read_md(state)
        if not image_dir.exists():
            self.logger.warning(f"文件 {md_path_obj.name} 暂无图片要处理")
            return {'md_content': md_content}

        # 2. 扫描图片 & 提取上下文
        image_list = self.scanner.scan_img_dir(
            image_dir, md_content,
            image_extensions=config.image_extensions,
            context_length=config.img_content_length,
        )

        # 3. VLM 生成摘要
        summaries = self.summarizer.summarize_all(
            document_title=md_path_obj.stem,
            image_list=image_list,
            vl_model=config.vl_model,
            requests_per_minute=config.requests_per_minute,
        )

        # 4. 上传 & 替换
        new_md_content = self.uploader.upload_and_replace(
            document_name=md_path_obj.stem,
            md_content=md_content,
            images_summaries=summaries,
            image_list=image_list,
            minio_bucket=config.minio_bucket,
            minio_base_url=config.get_minio_base_url(),
        )

        # 5. 备份
        self.file_handler.backup(md_path_obj, new_md_content)

        return {'md_content': new_md_content}


if __name__ == '__main__':
    setup_logging()
    md_path = r'E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto\万用表RS-12的使用.md'
    test_state = {
        'md_path': md_path,
        'md_content': Path(md_path).read_text(encoding='utf-8'),
        'file_title': '万用表RS-12的使用',
    }
    node = MarkDownImageNode()
    result = node(test_state)

    print("=" * 50)
    count_pattern = r'!\[.*?]\(.*?\)'
    print(f"共处理图片数: {len(re.findall(count_pattern, result['md_content']))}")
    print(f"结果总长度: {len(result['md_content'])} 字符")
