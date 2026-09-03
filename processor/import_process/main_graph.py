import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from processor.import_process.nodes.ducment_split import DocumentSplitNode
from processor.import_process.nodes.entry import EntryNode
from processor.import_process.nodes.import_milvus import ImportMilvusNode
from processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from processor.import_process.nodes.md_img import MarkDownImageNode
from processor.import_process.nodes.pdf_to_md import PdfToMdNode
from processor.import_process.state import ImportGraphState, create_default_state


# 定义路由函数,判断开始走哪个节点,啥都不是走END 节点
def import_router(state):
    if state.get('is_md_read_enabled'):
        return 'md_img_node'
    elif state.get('is_pdf_read_enabled'):
        return 'pdf_to_md_node'
    else:
        return END

# 创建 StateGraph，指定状态类型是 ImportGraphState
def create_import_graph():
    # 构建图对象
    graph_pipeline = StateGraph(ImportGraphState)
    # 添加节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }
    # 添加入口节点
    graph_pipeline.set_entry_point('entry_node')
    # 添加剩余节点
    for name, node in nodes.items():
        graph_pipeline.add_node(name, node)

    # 添加边：定义节点之间的执行顺序和条件路由,当开始和结束唯一时,可以不添加START和END边
    graph_pipeline.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "pdf_to_md_node":"pdf_to_md_node",
            "md_img_node":"md_img_node",
            END: END
        }
    )
    # 3.2 顺序边
    graph_pipeline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pipeline.add_edge("md_img_node", "document_split_node")
    graph_pipeline.add_edge("document_split_node", "item_name_rec_node")
    graph_pipeline.add_edge("item_name_rec_node", "bge_embedding_node")
    graph_pipeline.add_edge("bge_embedding_node", "import_milvus_node")
    graph_pipeline.add_edge("import_milvus_node", END)

    # 编译图：生成可执行的 CompiledStateGraph
    return graph_pipeline.compile()
# 创建全局图实例
kb_import_graph_app= create_import_graph()


# 启动工作流
def run_import_graph(import_file_path,file_dir):
    """
        便捷函数：运行导入流程

        Args:
            import_file_path: 输入文件路径（PDF 或 MD）
            file_dir: 本地工作目录

        Returns:
            最终状态字典
        """
    # 构建初始状态（entry_node 会根据真实文件后缀自行判断文件类型）
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir,
    }
    init_state = create_default_state(**state)

    # 调用stream获取每个节点的处理结果
    final_state = None
    for i in kb_import_graph_app.stream(init_state):
        for node_name,node_output in i.items():
            print(f'运行节点:{node_name}')
            final_state = node_output
    return  final_state


# 测试
if __name__ == '__main__':
    import_file_path = r'E:\11_尚硅谷AI全能开发技术之项目【掌柜智库】\2.资料\pdf文档\doc\万用表RS-12的使用.pdf'
    file_dir = r'E:\temp_dir'

    final_state = run_import_graph(
        import_file_path,
        file_dir
    )
    print(json.dumps(final_state, indent=4,ensure_ascii=False))

    # 打印图结构
    print('#'*50)
    kb_import_graph_app.get_graph().print_ascii()
