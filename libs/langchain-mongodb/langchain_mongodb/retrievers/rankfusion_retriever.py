"""Rank Fusion Retriever."""


from abc import ABC, abstractmethod
import asyncio
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pymongo.collection import Collection
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.pipelines import text_search_stage, vector_search_stage
from langchain_mongodb.utils import make_serializable

class ChildRetriever(ABC):
    @abstractmethod
    def name(self) -> str:
        """Unique name, used in scoreDetails and in pipeline labeling."""
        ...
    @abstractmethod
    def weight(self) -> float:
        """Weight for this retriever in fusion."""
        ...
    @abstractmethod
    async def to_pipeline(self, query: str,k:int) -> dict[str,Any]:
        """
        Return a sub-pipeline (as a list of aggregation stages or a pipeline fragment)
        that yields “_id” and “score” (or `meta: {score}`) fields for documents.
        May use either text query or vector query (or both), depending on the retriever.
        """
        ...

    def embedding_key(self) -> Optional[str]:
        """Return the vector embedding field name if applicable, else None."""
        return None

class FullTextRetriever(ChildRetriever):
    def __init__(self, name:str,search_index_name:str, weight:float=0.5,filter:dict[str,Any]=None,include_scores:bool = True):
        self._name = name
        self.search_index_name = search_index_name
        self._w = weight
        self.filter = filter
        self.include_scores = include_scores


    def name(self): 
        return self._name
    def weight(self): 
        return self._w

    async def to_pipeline(self, query,k):
        return text_search_stage(query=query,search_field=self.search_index_name,limit=k,filter=self.filter,include_scores=self.include_scores)

class VectorRetriever(ChildRetriever):
    def __init__(self, name:str, vectorstore:MongoDBAtlasVectorSearch, weight:float=0.5,pre_filter:dict[str,Any]=None,oversampling_factor:int = 10):
        self._name = name
        self.vectorstore = vectorstore
        self._w = weight
        self.pre_filter = pre_filter
        self.oversampling_factor = oversampling_factor

    def name(self): return self._name
    def weight(self): return self._w

    def embedding_key(self) -> Optional[str]:
        return self.vectorstore._embedding_key

    async def to_pipeline(self, query,k):
        query_vector = await self.vectorstore._embedding.aembed_query(query)
        return vector_search_stage(
                query_vector=query_vector,
                search_field=self.vectorstore._embedding_key,
                index_name=self.vectorstore._index_name,
                top_k=k,
                filter=self.pre_filter,
                oversampling_factor=self.oversampling_factor,
            )
    
class MongoDBRankFusionRetriever(BaseRetriever):

    collection: Collection
    """MongoDB Collection on an Atlas cluster"""

    search_pipelines: List[ChildRetriever]

    k: int = 4
    """Number of documents to return."""

    text_key: str = "text"
    """The main key to return as the documents page_content."""

    post_filter: Optional[List[Dict[str, Any]]] = None
    """(Optional) Pipeline of MongoDB aggregation stages for postprocessing."""
    show_embeddings: float = False
    """If true, returned Document metadata will include vectors."""
    score_details: bool = False

    def __init__(
        self,
        collection: Collection,
        search_pipelines: List[ChildRetriever],
        k: int = 4,
        text_key: str = "text",
        post_filter: Optional[List[Dict[str, Any]]] = None,
        show_embeddings: bool = False,
        score_details: bool = False,
        normalize_weights: bool = True,
    ):
        self.collection = collection
        self.search_pipelines = search_pipelines
        self.k = k
        self.text_key = text_key
        self.post_filter = post_filter
        self.show_embeddings = show_embeddings
        self.score_details = score_details
        self.normalize_weights = normalize_weights

        # --- validate & normalize weights once ---
        total_weight = sum(r.weight() for r in search_pipelines)
        if total_weight == 0:
            raise ValueError("Total retriever weight cannot be zero.")

        if total_weight > 1.0:
            if normalize_weights:
                for r in search_pipelines:
                    r._w = r.weight() / total_weight
                print(
                    f"[MongoDBRankFusionRetriever] Normalized retriever weights (sum was {total_weight:.2f})"
                )
            else:
                raise ValueError(
                    f"Total retriever weight {total_weight:.2f} > 1.0. "
                    "Either reduce weights or enable normalize_weights=True."
                )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Async implementation - this is the primary one."""
        pipeline: List[Any] = []
        
        # Build pipelines asynchronously (this is where the speedup happens)
        input_pipelines = await self._build_input_pipelines_async(query)

        # Assemble
        fusion_stage = {
            "$rankFusion": {
                "input": input_pipelines,
                "k": self.k,
                "scoreDetails": self.score_details,
            }
        }

        pipeline.append(fusion_stage)
        if not self.show_embeddings:
            embedding_keys = [
                r.embedding_key() for r in self.search_pipelines if r.embedding_key()
            ]
            if embedding_keys:
                pipeline.append({"$project": {key: 0 for key in embedding_keys}})

        if self.post_filter is not None:
            pipeline.extend(self.post_filter)

        # Execution (MongoDB operations are sync, but that's fine)
        cursor = self.collection.aggregate(pipeline)
        
        # Formatting
        docs = []
        for res in cursor:
            text = res.pop(self.text_key)
            make_serializable(res)
            docs.append(Document(page_content=text, metadata=res))
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Sync version delegates to async."""
        from langchain_core.runnables.config import run_in_executor
        
        return run_in_executor(
            None,
            self._aget_relevant_documents,
            query,
            run_manager=run_manager.get_async(),
            **kwargs
        )

    async def _build_input_pipelines_async(self, query: str) -> List[Dict[str, Any]]:
        """Async pipeline building - runs embeddings concurrently."""
        tasks = [
            retriever.to_pipeline(query=query, k=self.k)
            for retriever in self.search_pipelines
        ]
        pipeline_fragments = await asyncio.gather(*tasks)

        input_pipelines = [
            {
                "name": r.name(),
                "weight": r.weight(),
                "pipeline": fragment,
            }
            for r, fragment in zip(self.search_pipelines, pipeline_fragments)
        ]

        return input_pipelines