"""
  @Author:LiShuo
  @Time:2026/8/25
  @Desc:
"""
# System Prompt —— 角色设定，极简
ITEM_NAME_SYSTEM_PROMPT = "你是商品识别专家，只输出商品名称字符串。"

# User Prompt Template —— 包含文件标题 + 切片上下文 + 格式要求
ITEM_NAME_USER_PROMPT_TEMPLATE = """
请从以下信息中识别出商品名称与型号：
文件名：{file_title}

正文切片（用于辅助识别）：
{context}

要求：
1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称；
2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
3. 如果无法识别商品名称,请返回 UNKNOWN。
"""

