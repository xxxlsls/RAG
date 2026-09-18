import re
from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.config import get_config
from processor.import_process.exceptions import ValidationError

from processor.import_process.state import ImportGraphState
from utils.markdown_util import (
    MarkdownTableLinearizer,
    protect_image_syntax,
    restore_image_syntax,
)




class DocumentSplitNode(BaseNode):
    name:str = "document_split_node"

    # # 装箱算法
    # def _pack(self,units:list, max_length:int) ->list:
    #     #units：要装的"物品"列表，每个元素是一段文本（一个段落或一个句子）
    #     sep = '\n\n'  # 段落之间的分隔符（空行）
    #     chunks = []
    #     current = ''  #str类型
    #     for unit in units:
    #         # 算上分隔符再判断装不装得下
    #         new_length = len(current) + len(sep) + len(unit) if current else len(unit)
    #         if new_length <= max_length:
    #             current = current + sep + unit if current else unit
    #         else:
    #             if current:  # 箱子非空才封箱
    #                 chunks.append(current)
    #             current = unit
    #     if current:
    #         chunks.append(current)
    #     return chunks


    def process(self, state) -> dict:
        # 获取的数据并且进行校验:def _get_input_validation(state)
        md_content, file_title,max_content_length, min_content_length = self._get_input_validation(state)
        sections = self._split_by_headings(md_content, file_title)

        # 二次切分：只处理超长章节，切出来的小块用新列表收集（边遍历边改原列表会出问题）
        split_sections = []
        for section in sections:
            split_sections.extend(self.split_long_section(section, max_content_length))

        # 短章节合并：不足 min_content_length 的章节，和后面的章节合并（合并不超过 max）
        merge_sections = self.merge_short_section(split_sections, max_content_length, min_content_length)

        # 组装标准chunk结构
        chunks = []
        for section in merge_sections:
            body  = section['body']
            chunks.append({
                'title': section['title'],
                'content': f"## {section['title']}\n\n{body}",  # 标题 + 正文
                'file_title': file_title,
                'parent_title': section.get('parent_title', ''),  # 父标题（Milvus 入库需要）
            })

        # Step5: 统计日志
        self._log_summary(file_title, len(sections), chunks)
        # Step6: 备份 chunks.json
        self._backup_chunks(chunks, state.get('file_dir', ''))

        return {'chunks': chunks}




    def  _get_input_validation(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        """获取输入参数并预处理"""
        self.log_step("step1", "切分文档的参数校验以及获取...")

        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValidationError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length




    def _split_by_headings(self, md_content, file_title):
        self.log_step("step2", "根据标题进行切分...")

        # ① 状态变量
        in_fence = False  # 是否在代码围栏内
        body_lines = []  # 当前章节攒着的正文行
        sections = []  # 所有产出的章节
        current_level = 0  # 当前标题级别（1~6）
        current_title = ""  # 当前章节标题文字
        hierarchy = [""] * 7  # 标题栈：下标1~6各存一级"当前生效"的标题

        # ② 正则升级：两个捕获组 → group(1)=井号串，group(2)=标题文字
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # ③ 结账函数：把当前攒的章节存进 sections
        def _flush():
            nonlocal body_lines  # 要重新赋值 → 必须声明（只读的不需要）
            body = '\n'.join(body_lines).strip()
            if current_title or body:
                # 找父标题：从自己级别往上一层层向上，找第一个非空的
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                # 兜底：没有上级标题 → 自己当自己的父；连自己都没有 → 用文件标题
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    'title': current_title if current_title else file_title,
                    'body': body,
                    'file_title': file_title,
                    'parent_title': parent_title
                })
            body_lines = []  # 结账完清空，攒下一章

        # ④ 主流程：逐行遍历（循环体里只有三块平级代码）
        for line in md_content.split('\n'):
        
            # 块1：围栏开关：碰到围栏行 → 切换 in_fence 标志
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_fence = not in_fence
        
            # 块2：算 match（和围栏 if 平级！每行都要走）——围栏内不认标题
            match = heading_re.match(line) if not in_fence else None
        
            # 块3：标题行 → 结账开新章；正文行 → 攒着（和围栏 if 平级）
            if match:
                _flush()  # 先把上一章结账
                level = len(match.group(1))  # ## → 2
                current_level = level
                current_title = match.group(2).strip()  # 标题文字（捕获组，不再 lstrip）
                hierarchy[level] = current_title  # 本级就位
                for i in range(level + 1, 7):  # 清掉所有下级残留
                    hierarchy[i] = ""
            else:
                body_lines.append(line)  # 正文行 → 攒着
        
        # ⑤ 循环外：最后结一次账（最后一个章节后面没有新标题来触发）
        _flush()
        
        return sections

    def split_long_section(self, section, max_content_length):
        """超长章节二次切分,返回section列表(不超长就原样返回)"""
        title = section['title']
        body = section['body']
        # ① 表格线性化：if "<table>" in body → body = MarkdownTableLinearizer.process(body)
        #   （记得 from utils.markdown_util import MarkdownTableLinearizer）
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)

        #  ② 算标题预算：title_prefix = f"{title}\n\n"；total = len(title_prefix) + len(body)
        title_prefix = f"{title}\n\n"
        total = len(title_prefix) + len(body)

        #  ③ total <= max_content_length → return [section]（不用切）
        if total <= max_content_length:
            # 不超长就不用切，但必须带上已线性化的 body：
            # 上面已经付过 playwright 截图 + VLM 调用的成本，
            # 原来直接 return [section] 会把结果扔掉，短章节的 <table> 原样入库。
            return [self._with_body(section, body)]

        #  ④ body_length = max_content_length - len(title_prefix)；<=0 也 return [section]
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [self._with_body(section, body)]

        #  ⑤ RecursiveCharacterTextSplitter 切 body（separators 抄上面）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        )
        # 切分前先摘出图片语法换成短占位符：separators 末尾的 " " 和 ""（字符级兜底）
        # 会把三四百字符的 MinIO URL 拦腰砍断，导致 ![alt](url) 分成两半、图片永久丢失。
        protected_body, image_store = protect_image_syntax(body)
        chunks = text_splitter.split_text(protected_body)
        #  ⑥ 切出来 <=1 块 → return [section]
        if len(chunks) <= 1:
            return [self._with_body(section, body)]
        new_sections = []
        for i, chunk in enumerate(chunks):
        #  ⑦ 组装子片段：每块 {'title': f"{title}({i+1})", 'body': text,
            #        'file_title': section['file_title'], 'parent_title': section['pare
            new_sections.append({
                'title': f"{title}({i+1})",
                # 还原后单个 chunk 可能略超 max_content_length（图片语法本身比预算还长时）。
                # 刻意取舍：宁可超长也不能把图片砍断。下游 Milvus content 上限 65535，无风险。
                'body': restore_image_syntax(chunk, image_store),
                'file_title': section['file_title'],
                'parent_title': section['parent_title'],
            })
        return new_sections

    @staticmethod
    def _with_body(section, body):
        """返回 body 被替换掉的 section 副本，其余字段原样保留。

        刻意不就地改 section['body']：process() 里的 sections 列表还要传给
        _log_summary 统计"原始节数"，就地修改会让统计口径和实际产出对不上。
        body 没变化时直接返回原对象，避免无谓拷贝。
        """
        if section.get('body') == body:
            return section
        new_section = dict(section)
        new_section['body'] = body
        return new_section

    def merge_short_section(self, sections, max_content_length, min_content_length):
        """短章节合并：不足 min_content_length 的章节，和后面的章节合并（合并不超过 max）"""
        merge_sections = []
        pending = None
        for section in sections:
            content = section['body']
            if pending is not None:
                # 两个条件：同父标题？合后不超 max？（算上 '\n\n' 的 2 个字
                if section['parent_title'] == pending['parent_title'] and len(pending['body']) + len(content) + 2 <= max_content_length:
                    pending['body'] += '\n\n' + content
                    pending['title'] = f"{pending['title']} / {section['title']}"  # 更新标题
                    continue
                else:
                    merge_sections.append(pending)
                    pending =   None
            if len(content) <= min_content_length:
                pending = dict(section)
            else:
                merge_sections.append(section)
        if pending is not None:
            merge_sections.append(pending)
        return merge_sections

    def _log_summary(self, file_title, sections_count, chunks):
        """切分完成的统计摘要日志"""
        self.log_step("step5", f"文档「{file_title}」切分完成：{sections_count}节 → {len(chunks)}块")
        if not chunks:
            self.logger.warning("chunks 为空，跳过统计")
            return
        lengths = [len(c['content']) for c in chunks]
        total = sum(lengths)
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = total // len(lengths)
        self.logger.info(f"总{total}字 | 最长{max_len}/最短{min_len}/平均{avg_len}")

    def _backup_chunks(self, chunks, file_dir):
        """把 chunks 导出成 chunks.json 调试文件"""
        if not file_dir or not chunks:
            return
        import json
        import os
        try:
            out_path = os.path.join(file_dir, "chunks.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"chunks 已备份至: {out_path}")
        except Exception as e:
            self.logger.warning(f"chunks 备份失败: {e}")



if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto"
    }

    result = document_node(state)  # 用 __call__ 调用，顺带能看到"开始/完成"日志
    chunks = result['chunks']
    print(f"共 {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] {chunk['title']} | 长度 {len(chunk['content'])}")
        print(chunk['content'][:300])