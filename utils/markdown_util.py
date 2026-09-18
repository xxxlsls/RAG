import base64
import os
import re
from typing import List, Tuple
from bs4 import BeautifulSoup



# Markdown 图片语法：![alt](url) 或 ![alt](url "标题")
# url 可能含空格、中文、百分号转义，甚至未编码的裸括号（历史入库数据），
# 所以用「一层括号配对」而不是 [^)]* 来定位结尾，否则会在 -(KLVV 处提前截断。
_IMAGE_SYNTAX_PATTERN = re.compile(
    r'!\[([^\]\n]*)\]\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)'
)

_IMAGE_PLACEHOLDER_PREFIX = '%%KBIMG'
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r'%%KBIMG(\d+)%%')


def strip_image_syntax(text: str) -> str:
    """把 Markdown 图片语法降级成纯 alt 文本，专供「喂给模型」前调用。

    背景：md_img_node 会把图片替换成 ![VLM摘要](http://minio/bucket/文档名/图片名)，
    其中 URL 含大量百分号转义（%E5%8D%8E%E4%B8%BA...）和 64 位哈希文件名，
    长度常超过正文本身。这串字符对语义零贡献，却会：
      - 稀释 BGE-M3 稠密向量，把 chunk 往「无意义字符」方向拉偏，降低召回
      - 往稀疏向量塞入 %E5%8D%8E 之类垃圾 token，干扰关键词匹配
      - 挤占 Reranker 的 max_length 和 LLM 提示词的字符预算
    而 alt 部分是 VLM 生成的图片摘要，是真正的语义信息，必须保留。

    重要：只在「送进模型」前调用。写入 Milvus 的 content 和拼进答案提示词的
    content 必须保留完整图片语法，否则图片传不到前端。

    已知边界：只处理一层括号嵌套，形如 ![a](u((x))) 的双层嵌套不匹配（实际不会出现）。
    """
    if not text:
        return text
    return _IMAGE_SYNTAX_PATTERN.sub(lambda m: m.group(1).strip(), text)


def protect_image_syntax(text: str) -> Tuple[str, List[str]]:
    """把 Markdown 图片语法换成短占位符，返回 (占位后的文本, 图片原文列表)。

    为什么需要：RecursiveCharacterTextSplitter 的 separators 末尾是 " " 和 ""
    （字符级兜底），而 MinIO 图片 URL 动辄三四百字符、还可能含未编码的空格与括号。
    切分点一旦落在 URL 内部就会把 ![alt](url) 拦腰砍断：
        chunk A 结尾 -> "![摘要](http://192.168.10."
        chunk B 开头 -> "170:9000/knowledge-base-files/%E5%8D%8E...jpg)"
    两半都是废数据，图片永久丢失（前端找不到配对括号会直接跳过）。

    占位符刻意只用 % 字母 数字，不含 separators 里的任何分隔符
    （. ! ? ; 空格 换行 中文句号），因此不会被二次切分。

    还原请配套使用 restore_image_syntax。
    """
    if not text:
        return text, []

    store: List[str] = []

    def _replacer(match: re.Match) -> str:
        store.append(match.group(0))
        return f"{_IMAGE_PLACEHOLDER_PREFIX}{len(store) - 1}%%"

    return _IMAGE_SYNTAX_PATTERN.sub(_replacer, text), store


def restore_image_syntax(text: str, store: List[str]) -> str:
    """把占位符还原成图片语法原文，store 由 protect_image_syntax 返回。

    下标越界（理论上不会发生）时保留占位符原样，
    这样异常能从 chunks.json 备份文件里直接看出来，而不是静默变成空串。
    """
    if not text or not store:
        return text

    def _replacer(match: re.Match) -> str:
        index = int(match.group(1))
        return store[index] if 0 <= index < len(store) else match.group(0)

    return _IMAGE_PLACEHOLDER_PATTERN.sub(_replacer, text)


class MarkdownTableLinearizer:
    """
    解决：HTML复杂合并单元格、无表头KV表、左上角空置交叉表、原生MD表
    """

    HTML_TABLE_PATTERN = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    MD_TABLE_PATTERN = re.compile(
        r'((?:^[ \t]*\|.*\|[ \t]*\n)'
        r'(?:^[ \t]*\|[ \t]*[-:]+[-| :]*\|[ \t]*\n)'
        r'(?:^[ \t]*\|.*\|[ \t]*(?:\n|$))*)',
        re.MULTILINE
    )

    @classmethod
    def _needs_vlm(cls, html_content: str) -> bool:
        """复杂度嗅探：表格是否复杂到需要 VLM 兜底"""
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table:
            return False
        # 信号1：表中嵌表 —— 在"外层 table 内部"再找一个 table
        has_nested_table = soup.find("table").find("table") is not None
        # 信号2：超深跨行 —— 任意单元格的 rowspan 超过 10
        has_deep_rowspan = any(int(td.get("rowspan", 1)) > 10 for td in soup.find_all(["td","th"])
    )

        return has_nested_table or has_deep_rowspan

    @classmethod
    def _extract_with_vlm(cls, img_path: str) -> str:
        """使用 VLM 提取表格内容"""

        from processor.import_process.config import get_config
        config = get_config()
        # "读取截图并转为base64编码"
        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read())
        b64 = base64_image.decode("utf-8")
        # 提示词,保持和方案三的降维处理效果一致
        messages =[
            {
                'role':'user',
                'content':[
                    {'type':'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                    {'type':'text', 'text':'请将图片中的表格转译为一维自然语言描述。要求：每行数据独立成句，包含所有列信息，不要遗漏单元格内容。格式：- 列名1:值1，列名2:值2，...'},
                ]
            }
        ]
        from utils.client.ai_clients import AIClients
        response = AIClients.get_openai().chat.completions.create(
            model=config.vl_model,  # 从哪拿模型名？
            messages = messages,
            max_tokens=2000 ,  # 思考题1
        )
        return response.choices[0].message.content





    @classmethod
    def _html_to_image(cls, html_content: str) -> str:
        """把 HTML 表格渲染成 PNG 截图，返回临时文件路径"""
        import tempfile
        from playwright.sync_api import sync_playwright

        # ① 包装成完整页面：中文字体 + 边框样式，让 VLM 看得清格子
        full_html = f"""
        <html><head><meta charset="utf-8"><style>
            body {{ font-family: 'Microsoft YaHei', Arial; padding: 20px; }}
            table {{ border-collapse: collapse; }}
            td, th {{ border: 1px solid #333; padding: 6px 10px; }}
        </style></head><body>{html_content}</body></html>
        """
        # ② 建临时文件：???（提示：NamedTemporaryFile 两个参数，delete=?，用完记得 close）
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        out_path = tmp.name
        tmp.close()
        # ③ 无头浏览器：启动 → 喂 HTML → 截表格元素 → 关闭
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.set_content(full_html)
            page.locator("table").first.screenshot(path=out_path)
            browser.close()
        return out_path

    @classmethod
    def process(cls, content: str) -> str:
        if not content:
            return content

        if "<table" in content.lower():
            content = cls.HTML_TABLE_PATTERN.sub(cls._replace_html_table, content)

        if "|" in content:
            content = cls.MD_TABLE_PATTERN.sub(cls._replace_md_table, content)

        return content

    @classmethod
    def _replace_html_table(cls, match) -> str:
        html_content = match.group(0)

        # 智能路由：怪物表（表中嵌表/超深跨行）→ 方案四：截图 + VLM 看图转译
        if cls._needs_vlm(html_content):
            try:
                image_path = cls._html_to_image(html_content)
                try:
                    text = cls._extract_with_vlm(image_path)
                    # 前后包 \n\n，和 _grid_to_text 的返回格式对齐，防止和上下文文字粘连
                    return "\n\n" + text.strip() + "\n\n"
                finally:
                    os.remove(image_path)  # 无论 VLM 成败，截图文件必须删
            except Exception:
                pass  # VLM 失败 → 不炸，掉下去走方案三兜底

        # 方案三：确定性代码降维（矩阵投影 → 意图嗅探 → 语义重构）
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table: return html_content

        rows = table.find_all("tr")
        if not rows: return html_content

        # 嗅探 HTML 中是否使用了标准的 <th> 表头标签
        has_th = len(table.find_all("th")) > 0

        grid = []
        for _ in range(len(rows)):
            grid.append([])

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                # 使用 separator 保证 <br> 换行能变成空格，不会粘连
                text = cell.get_text(separator=" ", strip=True)

                for r in range(row_idx, row_idx + rowspan):
                    while len(grid) <= r: grid.append([])
                    while len(grid[r]) < col_idx + colspan: grid[r].append(None)
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = text
                col_idx += colspan

        return cls._grid_to_text(grid, is_md=False, has_th=has_th)

    @classmethod
    def _replace_md_table(cls, match) -> str:
        md_text = match.group(0).strip()
        lines = md_text.split('\n')
        grid = []
        for line in lines:
            if re.match(r'^[ \t]*\|[ \t\-|:]+\|[ \t]*$', line): continue
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            grid.append(cells)
        # Markdown 表格天生自带表头结构
        return cls._grid_to_text(grid, is_md=True, has_th=False)

    @classmethod
    def _grid_to_text(cls, grid: List[List[str]], is_md: bool, has_th: bool) -> str:
        if not grid or not grid[0]: return ""

        cols_count = max(len(r) for r in grid)
        # 补齐不规则行的列数，防越界
        for r in grid:
            while len(r) < cols_count: r.append("")

        is_header_row = False
        if is_md or has_th:
            is_header_row = True
        # 重点防御：如果左上角是空的，这绝对是一个交叉表头！绝不能当废数据丢弃！
        elif grid[0][0] == "":
            is_header_row = True
        # 一般大于两列的表格，第一行基本都是表头
        elif cols_count > 2:
            is_header_row = True

        res = []

        if not is_header_row and cols_count == 2:
            # 【策略 1：纯 K-V 表格】
            for r in grid:
                k, v = r[0], r[1]
                if k or v:
                    k_str = k if k else "未知属性"
                    v_str = v if v else "无"
                    res.append(f"- 【{k_str}】：{v_str}。")
        else:
            # 【策略 2：带有表头定义的标准/交叉表格】
            headers = grid[0]
            for r in grid[1:]:
                # 跳过完全空的数据行
                if not any(r): continue

                subject = r[0] if r[0] else "未知项目"
                subject_header = headers[0] if headers[0] else ""

                props = []
                for c in range(1, cols_count):
                    head = headers[c] if headers[c] else f"属性{c}"
                    val = r[c] if r[c] else ""

                    if val and val not in ('-', '/', '\\', '无'):
                        props.append(f"{head}为{val}")

                if props:
                    prop_str = "，".join(props)
                    if subject_header:
                        res.append(f"- 【{subject}】(对应{subject_header})：{prop_str}。")
                    else:
                        # 针对左上角为空的情况，隐藏对应关系描述
                        res.append(f"- 【{subject}】：{prop_str}。")
                else:
                    if subject != "未知项目":
                        res.append(f"- 【{subject}】")

        return "\n\n" + "\n".join(res) + "\n\n"

if __name__ == '__main__':
    test_html = """
    <table>
    <tr><td rowspan="12">直流电压</td><td>200mV</td><td>±(0.5%+2)</td></tr>
    <tr><td>2000mV</td><td>±(0.5%+2)</td></tr>
    <tr><td>20V</td><td>±(0.5%+2)</td></tr>
    </table>
    """
    test_md = f"# 万用表参数说明\n这是正文第一段。\n{test_html}\n这是正文第二段。"

    print("需要VLM兜底吗:", MarkdownTableLinearizer._needs_vlm(test_html))
    print("=" * 50)
    result = MarkdownTableLinearizer.process(test_md)
    print(result)
    print("=" * 50)
    print("转译成功" if "<table" not in result.lower() else "转译失败：仍有 <table> 残留")