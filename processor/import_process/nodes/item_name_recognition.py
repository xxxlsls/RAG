import json
import os
from typing import Dict, Optional, List, Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from processor.import_process.base import BaseNode
from processor.import_process.exceptions import StateFieldError, ValidationError
from processor.import_process.state import ImportGraphState
from prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.markdown_util import strip_image_syntax


class ItemNameRecognitionNode(BaseNode):
    name:str = "item_name_recognition_node"


    def _validate_state(self,state:ImportGraphState):
        """参数校验"""
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        # 3. 校验标题名
        if not file_title:
            raise StateFieldError(node_name=self.name,  field_name="file_title", expected_type=str)

        # 4. 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        # 5.获取商品名识别需要的chunk数
        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)
        # 6. 获取商品名识别上下文的最大字符数
        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size


    def _prepare_item_name_recognition_context(self, chunks: list, item_name_chunk_k: int, item_name_chunk_size: int):
        """数据准备,构建识别上下文"""
        total = 0
        final_context = []
        for index,chunk in enumerate(chunks[:item_name_chunk_k]):
            # 判断chunk类型
            if not isinstance(chunk, dict):
                continue
            # 剥离图片语法：一条 MinIO URL 动辄三四百字符，会迅速吃光
            # item_name_chunk_size 预算，导致后面真正带商品名线索的切片被 break 掉。
            # 同样兼作 None 防护：原来 chunk.get("content") 为 None 时，
            # f-string 会拼出字面量 "None" 送进 LLM。
            chunk_content = strip_image_syntax(chunk.get("content") or "")
            context = f'【切片】-{index}-{chunk_content}'

            # 先判断再加入，保证语义完整性
            if total + len(context) > item_name_chunk_size:
                break

            total += len(context)
            final_context.append(context)
        return '\n'.join(final_context)



    def _recognition_name(self, file_title: str, item_name_recognition_context: str) -> str:
        """LLM识别商品名称"""
        try:
            # 1. 获取LLM客户端（不需要JSON格式）
            client = AIClients.get_openai_llm(response_format=False)
            # 2. 构建提示词
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title,
                context=item_name_recognition_context
            )
            system_prompt = ITEM_NAME_SYSTEM_PROMPT

            # 3. LLM调用
            llm_response = client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 4. 获取并校验结果
            llm_result = llm_response.content.strip()

            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """生成稠密+稀疏向量
        Args:
            item_name: 商品名商品名向量化
        参数:
            item_name: 商品名
        返回:
            (向量列表, 附加信息字典)；失败时返回 (None, None)
        """
        try:
            # 1. 获取嵌入模型客户端
            bge_m3_client = AIClients.get_bge_m3_client()
            # 编码
            vector_result = bge_m3_client.encode_documents([item_name])

            # 2. 获取稠密向量
            dense_vector = vector_result['dense'][0].tolist()

            # 3. 获取稀疏向量（CSR 矩阵）
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]

            # 注意：indices 是 token_id，data 是权重，不要搞反！
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()

            sparse_vector = dict(zip(token_id, weight))
            return dense_vector, sparse_vector

        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None

        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None


    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        """向Milvus插入数据"""
        #  1.向量有效性检查
        if not dense_vector or not sparse_vector:
            self.logger.error(f"向量无效，插入Milvus失败: {file_title}, {item_name}")
            return
        # 2. 获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端获取失败: {e}")
            return
        # 3. 数据操作
        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)
        
            # 组装一条数据（集合存在与否都要插入）
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        
            # 插入
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        """定义结构（Schema）→ 定义索引 → 真正创建"""
        schema = milvus_client.create_schema()

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector",
                              index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector",
                              index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema, index_params=index_param
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        """填充商品名"""
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

    def _backup_chunks(self, state, chunks):
        """将回填 item_name 后的切片备份到 json 文件，给下个节点单元测试使用"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        os.makedirs(local_dir, exist_ok=True)
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            self.logger.info(f"chunks 已备份至: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


    def process(self, state):
        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建商品名识别上下文
        item_name_recognition_context = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                           self.config.item_name_collection)

        # 6. 回填item_name信息
        self._fill_item_name(item_name, state, chunks)

        # 7.备份，给下个节点准备下测试数据。
        self._backup_chunks(state,chunks)

        return state


if __name__ == '__main__':
    from processor.import_process.base import setup_logging
    setup_logging()

    # 1. 读取 chunks.json（课件06 导出的）
    chunk_json_path = r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:        chunk_content = json.load(f)

    # 2. 构建 state
    state = {
        "file_title": "万用表的使用",
        "file_dir": r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto",
        "chunks": chunk_content
    }

    # 3. 实例化节点并调用（用 __call__ 走异常包装）
    node = ItemNameRecognitionNode()
    result = node(state)

    # 4. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")