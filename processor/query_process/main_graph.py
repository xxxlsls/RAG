"""查询流程主图

使用 LangGraph 构建知识库查询工作流。

流程结构：
    __start__ → item_name_confirm →(条件路由)→ multi_search →(三路并行)→ join → rrf → rerank → answer_output → __end__
"""

"""
                                            +-----------+                                    
                                            | __start__ |                                    
                                            +-----------+                                    
                                                   *                                         
                                                   *                                         
                                                   *                                         
                                        +-------------------+                                
                                        | item_name_confirm |.                               
                                        +-------------------+ .......                        
                                            ...                      .......                 
                                           .                                ......           
                                         ..                                       .......    
                               +--------------+                                          ....
                               | multi_search |                                             .
                           ****+--------------+*****                                        .
                       ****            *            ****                                    .
                  *****                *                *****                               .
               ***                     *                     ***                            .
+------------------+       +-----------------------+       +----------------+               .
| search_embedding |       | search_embedding_hyde |       | web_search_mcp |               .
+------------------+***    +-----------------------+    ***+----------------+               .
                       *****           *           *****                                    .
                            *****      *      *****                                         .
                                 ***   *   ***                                              .
                                   +------+                                                 .
                                   | join |                                                 .
                                   +------+                                                 .
                                       *                                                    .
                                       *                                                    .
                                       *                                                    .
                                    +-----+                                                 .
                                    | rrf |                                                 .
                                    +-----+                                                 .
                                       *                                                    .
                                       *                                                    .
                                       *                                                    .
                                  +--------+                                             ....
                                  | rerank |                                      .......    
                                  +--------+                                ......           
                                            ***                      .......                 
                                               *              .......                        
                                                **        ....                               
                                          +---------------+                                  
                                          | answer_output |                                  
                                          +---------------+                                  
                                                   *                                         
                                                   *                                         
                                                   *                                         
                                              +---------+                                    
                                              | __end__ |                                    
                                              +---------+                                    

"""




from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from processor.query_process.nodes.answer_output import AnswerOutputNode
from processor.query_process.nodes.hyde_search import HyDeSearchNode
from processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode
from processor.query_process.nodes.rerank import RerankNode
from processor.query_process.nodes.rrf import RrfNode
from processor.query_process.nodes.vector_search import VectorSearchNode
from processor.query_process.nodes.web_search_mcp import McpSearchNode
from processor.query_process.state import QueryGraphState



load_dotenv()



def route_after_item_confirm(state: QueryGraphState) -> bool:
    """条件路由函数

    商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    return bool(state.get("answer"))



def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。
    """
    # ① 定义 LangGraph 工作流
    workflow = StateGraph(QueryGraphState)

    # ② 实例化节点（含两个虚拟节点）
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp":McpSearchNode(),
        "join":lambda x: {},
        "rrf":RrfNode(),
        "rerank": RerankNode(),
        "answer_output":AnswerOutputNode(),
    }

    # ③ 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # ④ 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # ⑤ 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )

    # ⑥ 多路搜索分发（并行执行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # ⑦ 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    # ⑧ 顺序边：join → rrf → rerank → answer_output → END
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    # ⑨ 返回编译后的图
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()

#打印图
query_app.get_graph().print_ascii()
