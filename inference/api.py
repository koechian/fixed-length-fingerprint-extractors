from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import cv2
import base64
from datetime import datetime

from fingerprint_matcher import Generator


class EmbeddingRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded fingerprint image")


class CompareRequest(BaseModel):
    """Request to compare two fingerprint images."""

    image1: str = Field(..., description="Base64-encoded first fingerprint image")
    image2: str = Field(..., description="Base64-encoded second fingerprint image")
    use_texture_only: bool = Field(
        False,
        description="Use concatenated texture+minutiae (False, recommended) "
        "or texture-only (True)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image1": "data:image/png;base64,iVBORw0KGgoAAAANS...",
                "image2": "data:image/png;base64,iVBORw0KGgoAAAANS...",
                "use_texture_only": False,
            }
        }


class CompareEmbeddingsRequest(BaseModel):
    """Request to compare two pre-computed embeddings."""

    embedding1: List[float] = Field(
        ..., description="First embedding vector (256-dimensional)"
    )
    embedding2: List[float] = Field(
        ..., description="Second embedding vector (256-dimensional)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "embedding1": [0.123, -0.456, 0.789, "..."],
                "embedding2": [0.234, -0.567, 0.890, "..."],
            }
        }


class EmbeddingResponse(BaseModel):
    """Response with extracted embedding."""

    texture_embedding: List[float] = Field(
        ..., description="256-dimensional texture embedding"
    )
    minutiae_embedding: Optional[List[float]] = Field(
        None, description="256-dimensional minutiae embedding (if requested)"
    )
    embedding_norm: float = Field(
        ..., description="L2 norm of the embedding (should be ~1.0)"
    )
    timestamp: str = Field(..., description="ISO timestamp of extraction")


class CompareResponse(BaseModel):
    """Response with comparison result."""

    similarity_score: float = Field(
        ..., description="Similarity score [-1, 1] (higher = more similar)"
    )
    interpretation: str = Field(..., description="Human-readable interpretation")
    timestamp: str = Field(..., description="ISO timestamp of comparison")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global matcher

    matcher = Generator(
        model_path="../trained_models",
        num_subjects=8000,
        embedding_dims=256,
        use_binarization=True,
        ridge_width=5.0,
    )
    print("Ready")
    yield


app = FastAPI(
    title="Fingerprint Matching API",
    description=(
        "REST API for fingerprint embedding extraction, comparison, "
        "and verification using pre-trained DeepPrint model"
    ),
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc",
    lifespan=lifespan,
)

matcher: Optional[Generator] = None

# @app.on_event("startup")
# async def startup_event():
#     global matcher
#     try:
#         matcher = Generator(
#             model_path="../trained_models",
#             num_subjects=8000,
#             embedding_dims=256,
#             use_binarization=True,
#             ridge_width=5.0,
#         )
#         print("Ready")
#     except Exception as e:
#         print(f"Failed to initialize: {e}")
#         traceback.print_exc()
#         matcher = None


@app.post("/extract", response_model=EmbeddingResponse)
async def extract_embedding(request: EmbeddingRequest):
    """
    Extract embedding from a fingerprint image.

    Accepts a base64-encoded fingerprint image and returns the concatenated
    texture and minutiae embeddings.
    """

    try:
        img_bytes = base64.b64decode(request.image)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Failed to decode image data")

        tex_emb, min_emb = matcher.extract_embedding(img)

        # Convert to lists for JSON serialization
        texture_list = tex_emb.tolist()
        minutiae_list = min_emb.tolist() if min_emb is not None else None

        # Calculate norm
        embedding_norm = float(np.linalg.norm(tex_emb))

        return EmbeddingResponse(
            texture_embedding=texture_list,
            minutiae_embedding=minutiae_list,
            embedding_norm=embedding_norm,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract embedding: {str(e)}"
        )


@app.post("/compare", response_model=CompareResponse)
async def compare_fingerprints(request: CompareRequest):
    """
    Compare two fingerprint images.

    Accepts two base64-encoded fingerprint images and returns
    a similarity score and interpretation.

    **Similarity Score Ranges (concatenated texture+minutiae):**
    - 1.5 - 2.0: Same finger (high confidence)
    - 1.2 - 1.5: Possible match (medium confidence)
    - 0.8 - 1.2: Uncertain (borderline)
    - 0.5 - 0.8: Different fingers (high confidence)
    - 0.0 - 0.5: Very different (very high confidence)

    Note: Range is [0, 2] because embeddings are concatenated without re-normalization.
    """
    if matcher is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Service unavailable."
        )

    try:
        # Compare fingerprints
        score = matcher.compare(request.image1, request.image2)

        # Interpret result (range [0, 2] for concatenated embeddings)
        if score >= 1.5:
            interpretation = "Same finger (high confidence)"
        elif score >= 1.2:
            interpretation = "Possible match (medium confidence)"
        elif score >= 0.8:
            interpretation = "Uncertain (borderline)"
        elif score >= 0.5:
            interpretation = "Different fingers (high confidence)"
        else:
            interpretation = "Very different (very high confidence)"

        return CompareResponse(
            similarity_score=float(score),
            interpretation=interpretation,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to compare fingerprints: {str(e)}"
        )


@app.post("/compare-embeddings", response_model=CompareResponse)
async def compare_embeddings(request: CompareEmbeddingsRequest):
    """
    Compare two pre-computed embeddings.

    Accepts two embedding vectors (from /extract endpoint) and returns
    a similarity score using cosine similarity (dot product of L2-normalized vectors).

    **Note:** Embeddings must be L2-normalized (norm ≈ 1.0).
    The /extract endpoint returns normalized embeddings by default.

    **Similarity Score Ranges:**
    - 0.85 - 1.0: Same finger (high confidence)
    - 0.60 - 0.85: Possible match (medium confidence)
    - 0.40 - 0.60: Uncertain (borderline)
    - 0.10 - 0.40: Different fingers (high confidence)
    - -1.0 - 0.10: Very different (very high confidence)
    """
    try:
        # Convert to numpy arrays
        emb1 = np.array(request.embedding1, dtype=np.float32)
        emb2 = np.array(request.embedding2, dtype=np.float32)

        # Validate dimensions
        if emb1.shape[0] != emb2.shape[0]:
            raise ValueError(
                f"Embedding dimensions don't match: {emb1.shape[0]} vs {emb2.shape[0]}"
            )

        # Check if embeddings are normalized (norm should be close to 1.0)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if abs(norm1 - 1.0) > 0.1 or abs(norm2 - 1.0) > 0.1:
            # Not normalized, normalize them
            emb1 = emb1 / norm1
            emb2 = emb2 / norm2

        # Compute cosine similarity (dot product of normalized vectors)
        score = float(np.dot(emb1, emb2))

        # Interpret result
        if score >= 0.85:
            interpretation = "Same finger (high confidence)"
        elif score >= 0.60:
            interpretation = "Possible match (medium confidence)"
        elif score >= 0.40:
            interpretation = "Uncertain (borderline)"
        elif score >= 0.10:
            interpretation = "Different fingers (high confidence)"
        else:
            interpretation = "Very different (very high confidence)"

        return CompareResponse(
            similarity_score=score,
            interpretation=interpretation,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to compare embeddings: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
