# API reference

The API is served at `http://127.0.0.1:8000`. Interactive OpenAPI
documentation is available at `/docs`. JSON is used unless an endpoint is
marked as `multipart/form-data` or NDJSON.

## Service and files

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/health` | Service and Qdrant availability |
| `GET` | `/keywords` | Supported COCO labels and keyword mappings |
| `GET` | `/images/{filename}` | Serve an image by its filename |
| `GET` | `/images/by-id/{image_id}` | Serve an indexed image by ID |
| `POST` | `/files/reveal` | Reveal a local indexed image; body: `{ "image_id": "..." }` |

## Search

| Method | Path | Request | Result |
| --- | --- | --- | --- |
| `POST` | `/search` | `{ "query": "cat on a sofa", "k": 10 }` | Semantic CLIP matches (`path`, `score`) |
| `POST` | `/search/class` | `{ "label": "cat", "k": 10 }` | YOLO object matches (`path`, `confidence`) |
| `POST` | `/search/unified` | `{ "query": "cat on a sofa", "labels": ["cat"], "k": 10 }` | Combined CLIP/YOLO matches |
| `POST` | `/search/face/embed` | Multipart field `file` | ArcFace query vector and detection score |
| `POST` | `/search/face` | `{ "embedding": [...], "k": 10, "threshold": 0.4 }` | Face matches (`path`, `score`) |
| `POST` | `/search/face/upload` | Multipart field `file`; optional `limit`, `k`, `threshold` | Upload-and-search convenience endpoint |

All JSON search requests accept optional `limit` for a bounded test scope.
`k` is between 1 and 50. Unified-search `labels` is optional: when omitted,
labels are inferred from the text query. A unified search may omit `query` when
at least one explicit object label is supplied; this performs object-only
search without CLIP.

When multiple object labels are selected, results matching more of those labels
are ranked first. Within the same object-match tier, a higher CLIP semantic
score promotes the image; YOLO confidence then breaks remaining ties. Results
that match only the text context follow object matches.

Progress-streaming variants return `application/x-ndjson`, one JSON object per
line:

| Method | Path |
| --- | --- |
| `POST` | `/search/stream` |
| `POST` | `/search/class/stream` |
| `POST` | `/search/unified/stream` |
| `POST` | `/search/face/stream` |

Use the same JSON body as the corresponding non-streaming endpoint.

## Indexing

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/index/reconcile` | Scan configured folders and reconcile the catalogue; returns a run ID |
| `POST` | `/index/run/full/{module}` | Start a full run for `clip`, `yolo`, or `faces` |
| `POST` | `/index/run/background` | Start a gap run for configured background modules |
| `GET` | `/index/status` | Active/latest run, module statistics, background schedule, and GPU status |
| `GET` | `/index/faces-ready` | Whether at least one indexed image has face data |

`/index/status` accepts `module=clip|yolo|faces` and `stats=true|false`.
Run-starting endpoints return `202 Accepted`; poll `/index/status` until the
run is complete. Conflicting runs return `409 Conflict`.

## Scan settings

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/settings/scan` | Read scan and background-indexer configuration |
| `PUT` | `/settings/scan` | Update supplied configuration fields |

Example update:

```json
{
  "include_directories": ["/absolute/path/to/photos"],
  "ignore_globs": ["**/.git/**"],
  "background_indexer_enabled": false,
  "schedule_interval_days": 7,
  "background_modules": ["clip", "yolo", "faces"]
}
```

## People

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/people` | List clustered people; query: `min_face_count`, `limit`, `offset` |
| `PATCH` | `/people/{person_id}` | Rename; body: `{ "name": "Name" }` |
| `GET` | `/people/{person_id}/faces` | List faces assigned to a person |
| `POST` | `/people/merge` | Merge people; body: `{ "person_ids": ["id1", "id2"] }` |
| `POST` | `/people/split` | Split a person by face IDs or groups |
| `GET` | `/people/low-confidence` | Low-confidence detections; query: `max_score`, `limit` |
| `GET` | `/people/cluster/status` | Latest clustering state |
| `POST` | `/people/cluster?regroup=false` | Start a clustering-only run |

For a split request, send either `{ "person_id": "...", "face_ids": ["..."] }`
or `{ "person_id": "...", "groups": [["..."], ["..."]] }`.

## Errors

Invalid requests return `400` or FastAPI validation error `422`. Missing
images, people, faces, or indexed search data commonly return `404`. A service
whose models or index executor have not finished starting returns `503`.
