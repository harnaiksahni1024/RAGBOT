
from pydantic import BaseModel,Field
from langchain.output_parsers import PydanticOutputParser
#create an outpuit parser to get a structured output

class QaResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The complete and factual answer to the user's question. "
                    "Do not make up information. Only answer based on the provided context. "
                    "If the context is insufficient, respond with 'I don't have enough context.'"
    )
    confidence: str = Field(
        ...,
        description="Confidence level of the answer based on the context (High/Medium/Low). "
                    "Do not guess if the context does not support the answer."
    )  


parser = PydanticOutputParser(pydantic_object=QaResponse)
