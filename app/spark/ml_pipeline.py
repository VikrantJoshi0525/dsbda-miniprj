"""
Spark MLlib Sentiment Classification Pipeline.

Builds a full ML pipeline:
    Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression

Supports training, evaluation, prediction, and model persistence.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    StringIndexer,
    IndexToString,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.types import DoubleType

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELS_DIR


# ══════════════════════════════════════════════════════════════════
# Data Classes for results
# ══════════════════════════════════════════════════════════════════

@dataclass
class ClassMetrics:
    """Per-class precision / recall / F1."""
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TrainResult:
    """Full training + evaluation result bundle."""
    accuracy: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    train_count: int
    test_count: int
    train_time_sec: float
    num_features: int
    reg_param: float
    max_iter: int
    class_metrics: list  # list[ClassMetrics]
    confusion_matrix: list  # list[dict] — rows of the confusion matrix
    label_order: list  # ordered label names
    feature_importance_top: list  # list[tuple[int, float]]  (feature_idx, coeff)
    model: Optional[PipelineModel] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════
# Pipeline Builder
# ══════════════════════════════════════════════════════════════════

def _build_pipeline(
    num_features: int = 10000,
    reg_param: float = 0.01,
    max_iter: int = 100,
    elastic_net: float = 0.0,
) -> Pipeline:
    """
    Build a Spark ML Pipeline for text classification.

    Stages:
        1. Tokenizer          — split clean_text into words
        2. StopWordsRemover   — remove English stop words
        3. HashingTF          — bag-of-words term frequencies
        4. IDF                — inverse document frequency weighting
        5. StringIndexer      — encode sentiment_label → numeric label
        6. LogisticRegression — multi-class classification
        7. IndexToString      — decode prediction back to label
    """
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")

    stopwords = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="filtered_tokens",
    )

    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=num_features,
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features",
        minDocFreq=3,
    )

    label_indexer = StringIndexer(
        inputCol="sentiment_label",
        outputCol="label",
        handleInvalid="keep",
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        regParam=reg_param,
        maxIter=max_iter,
        elasticNetParam=elastic_net,
        family="multinomial",
    )

    label_converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_label",
        labels=[],  # filled at fit time by the indexer model
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords,
        hashing_tf,
        idf,
        label_indexer,
        lr,
        # label_converter added post-fit
    ])

    return pipeline


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_sentiment_model(
    spark: SparkSession,
    df: DataFrame,
    test_ratio: float = 0.2,
    num_features: int = 10000,
    reg_param: float = 0.01,
    max_iter: int = 100,
    elastic_net: float = 0.0,
    seed: int = 42,
) -> TrainResult:
    """
    Train a Logistic Regression model on Spark DataFrame.

    Args:
        spark:        Active SparkSession.
        df:           Spark DataFrame with 'clean_text' and 'sentiment_label' cols.
        test_ratio:   Fraction held out for testing.
        num_features: HashingTF vocabulary size.
        reg_param:    L2 regularisation parameter.
        max_iter:     Max LR iterations.
        elastic_net:  ElasticNet mixing (0 = L2, 1 = L1).
        seed:         Random seed for reproducibility.

    Returns:
        TrainResult with all metrics, confusion matrix, and trained model.
    """
    # ── Prepare data ─────────────────────────────────────────
    # Keep only rows with clean_text and sentiment_label
    df = df.filter(
        F.col("clean_text").isNotNull()
        & (F.trim(F.col("clean_text")) != "")
        & F.col("sentiment_label").isNotNull()
    )

    # Binary: drop Neutral if very sparse (Sentiment140 has very few neutrals)
    label_counts = df.groupBy("sentiment_label").count().collect()
    labels_present = [row["sentiment_label"] for row in label_counts]

    # Train / Test split
    train_df, test_df = df.randomSplit([1.0 - test_ratio, test_ratio], seed=seed)
    train_df = train_df.cache()
    test_df = test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()

    # ── Build & fit pipeline ─────────────────────────────────
    pipeline = _build_pipeline(
        num_features=num_features,
        reg_param=reg_param,
        max_iter=max_iter,
        elastic_net=elastic_net,
    )

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0

    # ── Get label mapping from the StringIndexer stage ───────
    indexer_model = model.stages[4]  # StringIndexer is stage index 4
    label_order = list(indexer_model.labels)

    # ── Predict on test set ──────────────────────────────────
    predictions = model.transform(test_df)
    predictions = predictions.cache()

    # Decode numeric predictions back to label names
    converter = IndexToString(
        inputCol="prediction",
        outputCol="predicted_label",
        labels=label_order,
    )
    predictions = converter.transform(predictions)

    # ── Overall metrics ──────────────────────────────────────
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_wp = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_wr = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    evaluator_wf = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedFMeasure"
    )

    accuracy = evaluator_acc.evaluate(predictions)
    w_precision = evaluator_wp.evaluate(predictions)
    w_recall = evaluator_wr.evaluate(predictions)
    w_f1 = evaluator_wf.evaluate(predictions)

    # ── Per-class metrics ────────────────────────────────────
    class_metrics = []
    for idx, lbl in enumerate(label_order):
        # Filter to this class
        tp = predictions.filter(
            (F.col("label") == idx) & (F.col("prediction") == idx)
        ).count()
        fp = predictions.filter(
            (F.col("label") != idx) & (F.col("prediction") == idx)
        ).count()
        fn = predictions.filter(
            (F.col("label") == idx) & (F.col("prediction") != idx)
        ).count()
        support = predictions.filter(F.col("label") == idx).count()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        class_metrics.append(ClassMetrics(
            label=lbl,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            support=support,
        ))

    # ── Confusion matrix ─────────────────────────────────────
    cm_rows = []
    for i, actual_lbl in enumerate(label_order):
        row = {"Actual": actual_lbl}
        for j, pred_lbl in enumerate(label_order):
            count = predictions.filter(
                (F.col("label") == i) & (F.col("prediction") == j)
            ).count()
            row[f"Pred_{pred_lbl}"] = count
        cm_rows.append(row)

    # ── Feature importance (top coefficients) ────────────────
    lr_model = model.stages[5]  # LogisticRegression model
    feat_importance = []
    try:
        # For binary, coefficients is a Vector; for multinomial, it's a Matrix
        coeffs = lr_model.coefficientMatrix
        # Sum absolute coefficients across classes
        import numpy as np
        coeff_array = coeffs.toArray()
        abs_sum = np.abs(coeff_array).sum(axis=0)
        top_indices = abs_sum.argsort()[-20:][::-1]
        feat_importance = [(int(idx), round(float(abs_sum[idx]), 4)) for idx in top_indices]
    except Exception:
        pass

    # ── Cleanup cache ────────────────────────────────────────
    train_df.unpersist()
    test_df.unpersist()
    predictions.unpersist()

    return TrainResult(
        accuracy=round(accuracy, 4),
        weighted_precision=round(w_precision, 4),
        weighted_recall=round(w_recall, 4),
        weighted_f1=round(w_f1, 4),
        train_count=train_count,
        test_count=test_count,
        train_time_sec=round(train_time, 2),
        num_features=num_features,
        reg_param=reg_param,
        max_iter=max_iter,
        class_metrics=class_metrics,
        confusion_matrix=cm_rows,
        label_order=label_order,
        feature_importance_top=feat_importance,
        model=model,
    )


# ══════════════════════════════════════════════════════════════════
# Model Persistence
# ══════════════════════════════════════════════════════════════════

def save_model(model: PipelineModel, name: str = "lr_sentiment") -> str:
    """Save trained PipelineModel to disk."""
    path = str(MODELS_DIR / name)
    model.write().overwrite().save(path)
    return path


def load_model(spark: SparkSession, name: str = "lr_sentiment") -> PipelineModel:
    """Load a saved PipelineModel."""
    path = str(MODELS_DIR / name)
    return PipelineModel.load(path)
