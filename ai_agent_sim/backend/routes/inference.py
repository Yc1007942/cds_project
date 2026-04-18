"""
Inference API endpoints — AI/Human classification + engagement regression.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from ml_model import get_classifier, get_regressor, get_extractor

router = APIRouter()


class InferenceRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    label: str
    prediction: int
    confidence: float
    ai_prob: float
    human_prob: float
    word_count: int
    char_count: int
    features_used: int


class RegressionResponse(BaseModel):
    score: float
    agents_to_spawn: int
    word_count: int
    char_count: int


@router.post("/classify", response_model=ClassificationResponse)
async def classify_text(req: InferenceRequest):
    """Classify text as AI-generated or Human-written using BERT"""
    classifier = get_classifier()

    result = classifier.predict(req.text)

    return ClassificationResponse(
        label=result["label"],
        prediction=result.get("prediction", -1),
        confidence=result["confidence"],
        ai_prob=result["ai_prob"],
        human_prob=result["human_prob"],
        word_count=len(req.text.split()),
        char_count=len(req.text),
        features_used=768,  # BERT hidden size
    )


@router.post("/score", response_model=RegressionResponse)
async def predict_score(req: InferenceRequest):
    """Predict engagement score using the pre-trained regressor"""
    regressor = get_regressor()

    score = regressor.predict_score_from_text(req.text)
    agents = regressor.calculate_agents_to_spawn(score)

    return RegressionResponse(
        score=round(score, 2),
        agents_to_spawn=agents,
        word_count=len(req.text.split()),
        char_count=len(req.text),
    )


@router.get("/model-info")
async def model_info():
    """Get info about loaded models"""
    classifier = get_classifier()
    regressor = get_regressor()

    return {
        "classifier": {
            "loaded": classifier.model is not None,
            "features": 768,  # BERT hidden dimension
            "type": "BertForSequenceClassification" if classifier.model else None,
            "tokenizer_source": classifier.tokenizer_source,
            "ai_label_index": classifier.ai_label_index,
            "human_label_index": classifier.human_label_index,
            "label_names": classifier.label_names,
        },
        "regressor": {
            "loaded": regressor.model is not None,
            "features": len(regressor.feature_names) if regressor.feature_names else 783,
            "type": type(regressor.model).__name__ if regressor.model else None,
            "n_estimators": regressor.model.n_estimators if regressor.model and hasattr(regressor.model, 'n_estimators') else None,
        },
    }


@router.post("/extract-features")
async def extract_features_endpoint(req: InferenceRequest):
    """Extract and return the feature vector for a given text (for debugging/display)"""
    extractor = get_extractor()
    features_df = extractor.extract(req.text)

    # Return only non-embedding features for display
    display_features = {
        k: float(v) for k, v in features_df.iloc[0].to_dict().items()
        if not k.startswith('emb_')
    }

    return {"features": display_features}
