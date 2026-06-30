"""Custom Vast Serverless PyWorker.

Adds /v1/score and /v1/rerank routes (vLLM pooling / cross-encoder scoring) that the
stock `openai` worker does not expose, and forwards both verbatim to the local vLLM
OpenAI server. Autoscaler workload scales with the number of documents per request.

Deploy by pointing PYWORKER_REPO at this repository on a Vast vLLM serverless template.
"""
import glob
import os

from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE    = "/var/log/portal/vllm.log"
MODEL_HEALTHCHECK = "/health"
MODEL = os.environ.get("MODEL_NAME", "")
# Front images pre-staged on disk by the template onstart (file:// reads avoid base64 over the wire).
MEDIA_DIR = os.environ.get("MEDIA_DIR", "/workspace/photos")


def n_docs(payload):
    """Autoscaler cost ~ number of (query, document) pairs in the request."""
    docs = payload.get("documents")            # /v1/rerank schema
    if isinstance(docs, list):
        return float(max(1, len(docs)))
    text_2 = payload.get("text_2")             # /v1/score schema: text_1=query, text_2=[docs]
    if isinstance(text_2, list):
        return float(max(1, len(text_2)))
    return 1.0


def _img(path):
    return {"content": [{"type": "image_url", "image_url": {"url": "file://" + path}}]}


def score_benchmark():
    """Representative startup benchmark for the autoscaler. Uses pre-staged local images (1 query
    vs ~10 docs) so the perf estimate reflects the real multimodal workload; falls back to a tiny
    text-only request when no media is staged."""
    imgs = sorted(glob.glob(f"{MEDIA_DIR}/*/0.jpg"))[:11]
    if len(imgs) >= 2:
        return {"model": MODEL, "text_1": _img(imgs[0]), "text_2": [_img(p) for p in imgs[1:]]}
    return {"model": MODEL, "text_1": "example query", "text_2": ["document one", "document two"]}


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK,
    handlers=[
        HandlerConfig(
            route="/v1/score",
            allow_parallel_requests=True,
            max_queue_time=600.0,
            workload_calculator=n_docs,
            benchmark_config=BenchmarkConfig(generator=score_benchmark, runs=3, concurrency=4),
        ),
        HandlerConfig(
            route="/v1/rerank",
            allow_parallel_requests=True,
            max_queue_time=600.0,
            workload_calculator=n_docs,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        on_error=["INFO exited: vllm", "RuntimeError: Engine", "Traceback (most recent call last):"],
        on_info=['"message":"Download'],
    ),
)

Worker(worker_config).run()
