from pydantic import BaseModel, Field, model_validator

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


class ClassSearchRequest(BaseModel):
    label: str = Field(..., min_length=1, examples=["dog"])
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max number of images to scan (for testing)",
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Max number of matching photos to return",
    )


class ImageMatch(BaseModel):
    path: str
    score: float


class ClassImageMatch(BaseModel):
    path: str
    confidence: float


class SearchResponse(BaseModel):
    query : str
    matches : list[ImageMatch]


class ClassSearchResponse(BaseModel):
    label: str
    matches: list[ClassImageMatch]


class FaceSearchRequest(BaseModel):
    embedding: list[float] = Field(
        ...,
        min_length=1,
        description="L2-normalized ArcFace embedding from /search/face/embed",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max number of images to scan (for testing)",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max number of matching photos to return",
    )
    threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a face match",
    )


class FaceEmbedResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int
    detection_score: float


class FaceImageMatch(BaseModel):
    path: str
    score: float


class FaceSearchResponse(BaseModel):
    matches: list[FaceImageMatch]


class UnifiedSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["sunset with airplane"])
    labels: list[str] | None = Field(
        default=None,
        description="YOLO object labels; auto-detected from query when omitted",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max number of images in scope (for testing)",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max unified results to return after merging sources",
    )


class UnifiedMatchResponse(BaseModel):
    path: str
    clip_score: float | None = None
    yolo: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    rank_score: float = 0.0


class UnifiedSearchResponse(BaseModel):
    query: str
    labels: list[str]
    matches: list[UnifiedMatchResponse]


class ScanConfigResponse(BaseModel):
    include_directories: list[str]
    ignore_globs: list[str]
    background_indexer_enabled: bool
    schedule_interval_days: int
    background_modules: list[str]
    last_background_run_at: str | None = None


class ScanConfigUpdate(BaseModel):
    include_directories: list[str] | None = None
    ignore_globs: list[str] | None = None
    background_indexer_enabled: bool | None = None
    schedule_interval_days: int | None = Field(default=None, ge=1, le=365)
    background_modules: list[str] | None = None


class IndexRunResponse(BaseModel):
    id: str
    module: str
    mode: str
    status: str
    phase: str
    progress_done: int
    progress_total: int
    percent: int = 0
    started_at: str
    finished_at: str | None
    last_error: str | None


class ModuleRunStatus(BaseModel):
    active_run: IndexRunResponse | None = None
    latest_run: IndexRunResponse | None = None


class IndexStatusResponse(BaseModel):
    active_run: IndexRunResponse | None
    latest_run: IndexRunResponse | None
    modules: dict[str, dict[str, int]]
    module_runs: dict[str, ModuleRunStatus] = Field(default_factory=dict)
    background: dict
    scope_total: int
    gpu: dict = Field(default_factory=dict)


class FacesReadyResponse(BaseModel):
    ready: bool
    done: int
    total: int


class PersonSummaryResponse(BaseModel):
    id: str
    name: str | None
    display_name: str
    is_named: bool
    face_count: int
    preview_face_id: str | None = None
    preview_image_id: str | None = None
    sample_image_ids: list[str] = Field(default_factory=list)
    sample_image_paths: list[str] = Field(default_factory=list)


class PersonDetailResponse(PersonSummaryResponse):
    pass


class PersonListResponse(BaseModel):
    people: list[PersonSummaryResponse]
    total: int = 0
    total_all: int = 0
    returned: int = 0
    offset: int = 0
    min_face_count: int = 0


class PersonRenameRequest(BaseModel):
    name: str = Field(..., min_length=1)


class PersonMergeRequest(BaseModel):
    person_ids: list[str] = Field(..., min_length=2)


class PersonSplitRequest(BaseModel):
    person_id: str
    groups: list[list[str]] = Field(default_factory=list)
    face_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_faces(self) -> "PersonSplitRequest":
        if not any(self.groups) and not self.face_ids:
            raise ValueError("groups or face_ids required")
        return self

    def resolved_groups(self) -> list[list[str]]:
        if any(self.groups):
            return [group for group in self.groups if group]
        return [self.face_ids]


class RevealFileRequest(BaseModel):
    image_id: str = Field(..., min_length=1)


class PersonFaceResponse(BaseModel):
    face_id: str
    image_id: str
    image_path: str
    bbox: list[float]
    detection_score: float
