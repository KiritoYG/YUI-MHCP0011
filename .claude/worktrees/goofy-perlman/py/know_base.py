import asyncio
import httpx # 核心修复：使用异步 HTTP 客户端
from typing import List, Dict, Union
import json
import os
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from py.load_files import get_files_json
from py.get_setting import load_settings, base_path, KB_DIR
    
# --- Tiktoken 缓存设置（保留）---
def get_tiktoken_cache_path():
    cache_path = os.path.join(base_path, "tiktoken_cache")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

os.environ["TIKTOKEN_CACHE_DIR"] = get_tiktoken_cache_path()
# ---------------------------------

# --- 新增：清洗文本辅助函数 ---
def clean_text(text: str) -> str:
    """
    清洗文本，移除无法编码的 Unicode 代理字符（surrogates）。
    解决 'utf-8' codec can't encode character ... surrogates not allowed 错误。
    """
    if not isinstance(text, str):
        return str(text)
    # encode('utf-8', 'ignore') 会忽略掉非法的 surrogate 字符
    return text.encode('utf-8', 'ignore').decode('utf-8')


class MyOpenAICompatibleEmbeddings(Embeddings):
    """
    OpenAI 兼容的词嵌入类，使用 httpx 异步客户端进行非阻塞网络请求。
    """
    def __init__(self, base_url: str, model: str, api_key: str = "empty"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        # 假设 base_url 已经是 http://127.0.0.1:8000/minilm
        self.endpoint = f"{self.base_url}/embeddings"

    # --- 异步核心方法 ---
    async def _aembed(self, texts: Union[str, List[str]]) -> List[Dict]:
        """异步发送嵌入请求并处理响应"""
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        json_data = {"model": self.model, "input": texts}
        
        # 使用 httpx.AsyncClient 发送请求
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                # 调用词嵌入接口
                response = await client.post(self.endpoint, headers=headers, json=json_data)
                
                # 检查 HTTP 状态码
                response.raise_for_status() 
                
                return response.json()["data"]
                
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get('detail', e.response.text) if e.response.text else 'Unknown error'
                raise RuntimeError(f"Embedding API HTTP Error {e.response.status_code}: {detail}")
            except Exception as e:
                raise ConnectionError(f"Embedding API connection failed: {e.__class__.__name__}: {e}")

    # --- LangChain 兼容的同步方法 ---
    def embed_query(self, text: str) -> List[float]:
        data = asyncio.run(self.aembed_query(text))
        return data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = asyncio.run(self.aembed_documents(texts))
        return data

    # --- 暴露异步 LangChain 方法 ---
    async def aembed_query(self, text: str) -> List[float]:
        data = await self._aembed(text)
        return data[0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        data = await self._aembed(texts)
        return [r["embedding"] for r in data]


def chunk_documents(results: List[Dict], cur_kb) -> List[Document]:
    """为每个文件单独分块并添加元数据"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cur_kb["chunk_size"],
        chunk_overlap=cur_kb["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "!", "?", "."]
    )
    
    all_docs = []
    for doc in results:
        # 在分块前也可以简单清洗一下，防止 text_splitter 报错
        clean_content = clean_text(doc["content"])
        chunks = text_splitter.split_text(clean_content)
        for chunk in chunks:
            all_docs.append(Document(
                page_content=chunk,
                metadata={
                    "file_path": doc["file_path"],
                    "file_name": doc["file_name"],
                    "doc_id": f"{doc['file_path']}_{len(all_docs)}" 
                }
            ))
    return all_docs

# 核心修改：增加容错和数据清洗
async def build_vector_store(docs: List[Document], kb_id, cur_kb: Dict, cur_vendor: str):
    """构建并保存双索引"""
    if not isinstance(docs, list) or not all(isinstance(d, Document) for d in docs):
        raise ValueError("Input must be a list of Document objects")
    
    kb_dir = Path(KB_DIR)
    kb_dir.mkdir(parents=True, exist_ok=True)
    save_dir = kb_dir / str(kb_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========== BM25索引构建 (容错版) ==========
    try:
        bm25_path = save_dir / "bm25_index.json"
        
        if not docs:
            print("Warning: No documents provided for BM25.")
        else:
            # 1. 清洗数据，防止 Unicode 错误
            clean_docs_data = []
            for doc in docs:
                clean_metadata = {
                    k: clean_text(v) if isinstance(v, str) else v 
                    for k, v in doc.metadata.items()
                }
                clean_docs_data.append({
                    "page_content": clean_text(doc.page_content),
                    "metadata": clean_metadata
                })

            # 2. 保存 (使用 clean_docs_data)
            await asyncio.to_thread(
                lambda: json.dump(
                    {"docs": clean_docs_data}, 
                    open(bm25_path, "w", encoding="utf-8", errors="ignore"), 
                    ensure_ascii=False
                )
            )
            print(f"BM25 index saved successfully for KB {kb_id}")

    except Exception as e:
        # 即使 BM25 失败，也只打印警告，不中断程序
        print(f"⚠️ BM25 Index failed (Skipping): {str(e)}")
        # 尝试清理可能损坏的文件
        if 'bm25_path' in locals() and bm25_path.exists():
            try:
                os.remove(bm25_path)
            except:
                pass

    # ========== 向量索引构建 (使用异步客户端) ==========
    try:
        embeddings = MyOpenAICompatibleEmbeddings(
            model=cur_kb["model"],
            api_key=cur_kb["api_key"],
            base_url=cur_kb["base_url"],
        )
        
        batch_size = 20 
        vector_db = None
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            
            # 使用 asyncio.to_thread 运行同步的 FAISS 方法
            if vector_db is None:
                vector_db = await asyncio.to_thread(FAISS.from_documents, batch, embeddings)
            else:
                await asyncio.to_thread(vector_db.add_documents, batch)
            
            print(f"Processed {min(i+batch_size, len(docs))}/{len(docs)} documents")
        
        # 最终保存
        if vector_db:
            await asyncio.to_thread(vector_db.save_local, folder_path=str(save_dir), index_name="index")
            print(f"Vector store saved successfully for KB {kb_id}")
        
    except Exception as e:
        raise RuntimeError(f"Vector store build failed: {str(e)}")


async def load_retrievers(kb_id, cur_kb, cur_vendor):
    """加载双检索器 (带 BM25 缺失的回退机制)"""
    kb_path = Path(KB_DIR) / str(kb_id)
    bm25_path = kb_path / "bm25_index.json"
    
    # 1. 尝试加载 BM25
    bm25_retriever = None
    try:
        if bm25_path.exists():
            bm25_data = await asyncio.to_thread(json.load, open(bm25_path, "r", encoding="utf-8"))
            bm25_docs = [
                Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
                for doc in bm25_data["docs"]
            ]
            if bm25_docs:
                bm25_retriever = await asyncio.to_thread(BM25Retriever.from_documents, bm25_docs)
                bm25_retriever.k = cur_kb["chunk_k"]
    except Exception as e:
        print(f"Error loading BM25 (will fallback): {e}")

    # 2. 加载向量检索器
    embeddings = MyOpenAICompatibleEmbeddings(
        model=cur_kb["model"],
        api_key=cur_kb["api_key"],
        base_url=cur_kb["base_url"],
    )
    
    vector_db = await asyncio.to_thread(
        FAISS.load_local,
        folder_path=str(kb_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="index"
    )
    vector_retriever = vector_db.as_retriever(
        search_kwargs={"k": cur_kb["chunk_k"]}
    )

    # 3. 如果 BM25 加载失败（比如之前构建时跳过了），使用向量检索器顶替
    # 这样 EnsembleRetriever 相当于用了两个 VectorRetriever，不会报错
    if bm25_retriever is None:
        print("Fallback: Using Vector Retriever for BM25 slot.")
        bm25_retriever = vector_retriever

    return bm25_retriever, vector_retriever

async def query_vector_store(query: str, kb_id, cur_kb, cur_vendor):
    """使用EnsembleRetriever的混合查询"""
    bm25_retriever, vector_retriever = await load_retrievers(kb_id, cur_kb, cur_vendor)
    if "weight" not in cur_kb:
        cur_kb["weight"] = 0.5
        
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - cur_kb["weight"], cur_kb["weight"]],
    )
    
    # EnsembleRetriever.invoke 是同步阻塞的，需要放在线程中运行
    docs = await asyncio.to_thread(ensemble_retriever.invoke, query)
    
    # 格式转换
    return [{
        "content": doc.page_content,
        "metadata": doc.metadata,
    } for doc in docs]


async def process_knowledge_base(kb_id):
    """异步处理知识库的完整流程"""
    settings = await load_settings()
    cur_kb = None
    providerId = None
    for kb in settings["knowledgeBases"]:
        if kb["id"] == kb_id:
            cur_kb = kb
            providerId = kb["providerId"]
            break
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    
    if not cur_kb:
        raise ValueError(f"Knowledge base {kb_id} not found in settings")
        
    processed_results = await get_files_json(cur_kb["files"])
    
    chunks = chunk_documents(processed_results, cur_kb)
    
    # 调用异步版本的 build_vector_store
    await build_vector_store(chunks, kb_id, cur_kb, cur_vendor)

    return "知识库处理完成"

async def query_knowledge_base(kb_id, query: str):
    """查询知识库"""
    settings = await load_settings()
    cur_kb = None
    providerId = None
    for kb in settings["knowledgeBases"]:
        if kb["id"] == kb_id:
            cur_kb = kb
            providerId = kb["providerId"]
            break
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    
    if not cur_kb:
        return f"Knowledge base {kb_id} not found in settings"
        
    # 调用异步版本的 query_vector_store
    results = await query_vector_store(query, kb_id, cur_kb, cur_vendor)
    return results

async def rerank_knowledge_base(query: str , docs: List[Dict]) -> List[Dict]:
    settings = await load_settings()
    providerId = settings["KBSettings"]["selectedProvider"]
    cur_vendor = None
    for provider in settings["modelProviders"]:
        if provider["id"] == providerId:
            cur_vendor = provider["vendor"]
            break
    if cur_vendor == "jina":
        jina_api_key = settings["KBSettings"]["api_key"]
        model_name = settings["KBSettings"]["model"]
        top_n = settings["KBSettings"]["top_n"]
        documents = [doc.get("content", "") for doc in docs]
        url = settings["KBSettings"]["base_url"] + "/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jina_api_key}"
        }
        data = {
            "model": model_name,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            "return_documents": False
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Jina reranking failed: {response.text}")
        result = response.json()
        ranked_indices = [item['index'] for item in result.get('results', [])]
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
    elif cur_vendor == "Vllm":
        model_name = settings["KBSettings"]["model"]
        top_n = settings["KBSettings"]["top_n"]
        documents = [doc.get("content", "") for doc in docs]
        url = settings["KBSettings"]["base_url"] + "/rerank"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {
            "model": model_name,
            "query": query,
            "top_n": top_n,
            "documents": documents,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Vllm reranking failed: {response.text}")
        result = response.json()
        ranked_indices = [item['index'] for item in result.get('results', [])]
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
    else:
        return docs

kb_tool = {
    "type": "function",
    "function": {
        "name": "query_knowledge_base",
        "description": f"通过自然语言获取的对应ID的知识库信息。回答时，在回答的最下方给出信息来源。以链接的形式给出信息来源，格式为：[file_name](file_path)。file_path可以是外部资源，也可以是127.0.0.1上的资源。返回链接时，不要让()内出现空格。如果需要实现引用位置到跳转脚注链接的功能，请用句末用`[^1]`加脚注用`[^1]: [file_name](file_path)`的markdown语法。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "需要搜索的问题。",
                },
                "kb_id": {
                    "type": "string",
                    "description": "知识库的ID。"
                }
            },
            "required": ["kb_id","query"],
        },
    },
}