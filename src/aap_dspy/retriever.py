from aap_core.retriever import BaseRetriever
from aap_core.types import AgentMessage
from pydantic import ConfigDict, Field, field_validator

from dspy import Retrieve, Embeddings


class RetrieverAdapter(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: Retrieve | Embeddings = Field(
        ..., description="The dspy's retriever to use"
    )
    data_key: str = Field(
        default="context.data", description="The key to the data in the message"
    )

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("data_key must start with 'context.'")
        return v

    def retrieve(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if isinstance(self.retriever, Embeddings):
            results = self.retriever(message.query).passages
        else:
            results = self.retriever(message.query)

        data = []
        for result in results:
            data.append(result)

        data_key = self.data_key.replace("context.", "")
        content = " ".join(data) if len(data) > 1 else data
        if message.context is None:
            message.context = {data_key: content}
        else:
            message.context[data_key] = content
        return message
