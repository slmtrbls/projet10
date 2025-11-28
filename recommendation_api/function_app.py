import json
import logging
import os
from pathlib import Path
import pickle
from typing import List, Optional

import azure.functions as func
import numpy as np
import pandas as pd
from azure.storage.blob import BlobClient


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

TMP_DIR = Path("/tmp/recommendation")
EMBED_PATH = TMP_DIR / "embeddings_pca.pkl"
MODEL_PATH = TMP_DIR / "svd_best_model_pca.pkl"

MODELS_CONTAINER = os.getenv("MODELS_CONTAINER", "models")
EMBED_BLOB_NAME = os.getenv("EMBEDDINGS_BLOB", "embeddings_pca.pkl")
MODEL_BLOB_NAME = os.getenv("MODEL_BLOB", "svd_best_model_pca.pkl")

MAX_CANDIDATES = 20000
DEFAULT_TOP_K = 5

CACHED_EMBEDDINGS: Optional[pd.DataFrame] = None
CACHED_MODEL = None
CACHED_ITEMS: Optional[np.ndarray] = None


def _get_connection_string() -> str:
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str:
        raise RuntimeError("AzureWebJobsStorage n'est pas configurée.")
    return conn_str


def _download_blob(blob_name: str, destination: Path) -> None:
    logging.info("Téléchargement du blob '%s/%s'.", MODELS_CONTAINER, blob_name)
    blob_client = BlobClient.from_connection_string(
        conn_str=_get_connection_string(),
        container_name=MODELS_CONTAINER,
        blob_name=blob_name,
    )
    data = blob_client.download_blob().readall()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def _load_artifacts(force: bool = False) -> None:
    global CACHED_EMBEDDINGS, CACHED_MODEL

    if force or CACHED_EMBEDDINGS is None:
        _download_blob(EMBED_BLOB_NAME, EMBED_PATH)
        CACHED_EMBEDDINGS = pd.read_pickle(EMBED_PATH)
        logging.info("Embeddings chargés : %d items.", len(CACHED_EMBEDDINGS))

    if force or CACHED_MODEL is None:
        _download_blob(MODEL_BLOB_NAME, MODEL_PATH)
        with MODEL_PATH.open("rb") as handle:
            bundle = pickle.load(handle)
        CACHED_MODEL = bundle.get("model")
        if CACHED_MODEL is None:
            raise ValueError("Le pickle ne contient pas la clé 'model'.")
        logging.info("Modèle SVD chargé.")


def _ensure_candidates() -> np.ndarray:
    global CACHED_ITEMS
    if CACHED_ITEMS is None:
        assert CACHED_EMBEDDINGS is not None
        items = CACHED_EMBEDDINGS.index.to_numpy(dtype=np.int64)
        if items.size > MAX_CANDIDATES:
            items = items[:MAX_CANDIDATES]
        CACHED_ITEMS = items
        logging.info("Liste de %d items candidats prête.", items.size)
    return CACHED_ITEMS


def _parse_user_id(req: func.HttpRequest) -> int:
    user_id = req.params.get("user_id")
    if user_id is None:
        try:
            payload = req.get_json()
        except ValueError:
            payload = {}
        user_id = payload.get("user_id")

    if user_id is None:
        raise ValueError("Paramètre 'user_id' manquant.")

    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("Le paramètre 'user_id' doit être un entier.") from exc


def _parse_top_k(req: func.HttpRequest) -> int:
    top_k = req.params.get("top_k")
    if top_k is None:
        try:
            payload = req.get_json()
        except ValueError:
            payload = {}
        top_k = payload.get("top_k")

    if top_k is None:
        return DEFAULT_TOP_K

    try:
        value = int(top_k)
        return value if value > 0 else DEFAULT_TOP_K
    except (TypeError, ValueError):
        return DEFAULT_TOP_K


def _recommend_for_user(user_id: int, top_k: int) -> List[int]:
    assert CACHED_MODEL is not None
    candidates = _ensure_candidates()

    scores: List[tuple[int, float]] = []
    for item_id in candidates:
        try:
            estimate = CACHED_MODEL.predict(user_id, int(item_id)).est
        except Exception:
            continue
        scores.append((int(item_id), float(estimate)))

    if not scores:
        return []

    scores.sort(key=lambda tup: tup[1], reverse=True)
    return [item for item, _ in scores[:top_k]]


@app.on_startup()
def warmup(context: func.Context) -> None:
    logging.info("Initialisation au démarrage : pré-chargement des artefacts.")
    _load_artifacts(force=True)


@app.function_name(name="recommend")
@app.route(route="recommendations", methods=["GET", "POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Requête reçue pour la génération de recommandations.")
    try:
        _load_artifacts()
        user_id = _parse_user_id(req)
        top_k = _parse_top_k(req)
        recommendations = _recommend_for_user(user_id, top_k)
        payload = {"user_id": user_id, "top_k": top_k, "recommendations": recommendations}
        return func.HttpResponse(json.dumps(payload), mimetype="application/json")
    except ValueError as exc:
        logging.warning("Erreur de validation : %s", exc)
        return func.HttpResponse(str(exc), status_code=400)
    except Exception:
        logging.exception("Erreur inattendue.")
        return func.HttpResponse("Erreur interne.", status_code=500)

