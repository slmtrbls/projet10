import json
import logging
from pathlib import Path
import pickle
from typing import List, Optional

import azure.functions as func
import numpy as np
import pandas as pd


TMP_DIR = Path("/tmp")
EMBED_PATH = TMP_DIR / "embeddings_pca.pkl"
MODEL_PATH = TMP_DIR / "svd_best_model_pca.pkl"

# Caches chargés au premier appel (après cold start)
CACHED_EMBEDDINGS: Optional[pd.DataFrame] = None
CACHED_MODEL = None
CACHED_ITEMS: Optional[np.ndarray] = None

# Nombre maximum d'articles scorés (limite pour contenir les temps de réponse)
MAX_CANDIDATES = 20000
DEFAULT_TOP_K = 5 # Nombre de recommandations à générer


def _ensure_artifacts(
    embedding_blob: func.InputStream, model_blob: func.InputStream
) -> None:
    """
    Charge les blobs d'embeddings et de modèle dans /tmp puis en mémoire
    (opération effectuée une seule fois tant que l'instance reste chaude).
    """
    global CACHED_EMBEDDINGS, CACHED_MODEL

    if CACHED_EMBEDDINGS is None:
        logging.info("Chargement des embeddings PCA depuis le blob …")
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        EMBED_PATH.write_bytes(embedding_blob.read())
        CACHED_EMBEDDINGS = pd.read_pickle(EMBED_PATH)
        logging.info("Embeddings chargés : %d items.", len(CACHED_EMBEDDINGS))

    if CACHED_MODEL is None:
        logging.info("Chargement du modèle SVD depuis le blob …")
        MODEL_PATH.write_bytes(model_blob.read())
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        CACHED_MODEL = bundle.get("model")
        if CACHED_MODEL is None:
            raise ValueError("Le pickle ne contient pas la clé 'model'.")
        logging.info("Modèle SVD chargé.")


def _ensure_candidates() -> np.ndarray:
    """
    Prépare la liste des items candidats à scorer.
    On limite à MAX_CANDIDATES pour éviter un temps de réponse trop long.
    """
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
    """
    Récupère l'identifiant utilisateur depuis la query string ou le corps JSON.
    """
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
    """
    Permet de configurer top_k depuis la requête (facultatif).
    """
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
    """
    Applique le modèle SVD sur la liste d'items candidats.
    """
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


def main(
    req: func.HttpRequest,
    embedding_blob: func.InputStream,
    model_blob: func.InputStream,
) -> func.HttpResponse:
    logging.info("Requête reçue pour la génération de recommandations.")
    try:
        _ensure_artifacts(embedding_blob, model_blob)
        user_id = _parse_user_id(req)
        top_k = _parse_top_k(req)
        recommendations = _recommend_for_user(user_id, top_k)
        payload = {"user_id": user_id, "top_k": top_k, "recommendations": recommendations}
        return func.HttpResponse(json.dumps(payload), mimetype="application/json")
    except ValueError as exc:
        logging.warning("Erreur de validation : %s", exc)
        return func.HttpResponse(str(exc), status_code=400)
    except Exception as exc:  # pragma: no cover - logs serve production debugging
        logging.exception("Erreur inattendue.")
        return func.HttpResponse("Erreur interne.", status_code=500)

