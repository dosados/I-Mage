from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["кот на диване"])
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max number of images to search (for testing)",
    )
    k: int = Field(
        default = 1,
        ge = 1,
        le = 50,
        description="Amount of top-match photos to return"
    )


class ImageMatch(BaseModel):
    path: str
    score: float


class SearchResponse(BaseModel):
    query : str
    matches : list[ImageMatch]
