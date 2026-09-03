import json
from pathlib import Path

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.config import get_config
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState
from utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    name:str= "bge_embedding_node"


    def process(self,state:ImportGraphState):
        # 验证数据
        self.log_step("step1", "参数校验")
        chunks, config = self._validate_get_inputs(state)
        # 拿到批量嵌入的间隔值,防止gpu压力过大
        embedding_batch_size = config.embedding_batch_size

        total_length = len(chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_size):
            batch = chunks[i:i + embedding_batch_size]
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)
        state['chunks'] = final_chunks

        return state




    def _validate_get_inputs(self, state: ImportGraphState):
        """验证输入参数"""
        config = get_config()

        chunks = state.get('chunks')

        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)

        self.logger.info(f"嵌入的块数：{len(chunks)}")
        return chunks, config

    def _process_batch_chunks(self, batch, i, total_length):
        self.log_step("step2", f"处理批量块: {i + 1}-{i + len(batch)}/{total_length}")

        # 1. 拼接本批所有 chunk 的嵌入内容：item_name + "\n" + content
        embedding_contents = []
        for chunk in batch:
            item_name = chunk.get('item_name', '')
            content = chunk.get('content', '')
            embedding_contents.append(f"{item_name}\n{content}")

        # 2. 调 BGE-M3 批量嵌入（try 包住，失败 return batch 不炸）
        try:
            bge_m3_model = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)
        except Exception as e:
            self.logger.warning(f"嵌入失败: {e}")
            return batch

        # 3. 遍历本批每个 chunk，注入向量
        for index, chunk in enumerate(batch):
            # 3.1 稠密向量：embedding_result['dense'][index].tolist()
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 稀疏向量：CSR 按 indptr[index]~indptr[index+1] 拆
            csr_array = embedding_result['sparse']
            ind_ptr = csr_array.indptr
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))

            # 3.3 注入 chunk['dense_vector'] / chunk['sparse_vector']
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理chunk嵌入:批次{i + 1}-{i + len(batch)}/{total_length}")
        return batch

if __name__ == "__main__":
    setup_logging()

    base_temp_dir = Path(r"E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用\auto")

    input_path = base_temp_dir / "chunks_item_name.json"
    output_path = base_temp_dir / "chunks_item_name_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")
