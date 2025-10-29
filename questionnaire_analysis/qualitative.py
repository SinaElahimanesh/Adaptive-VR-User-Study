from __future__ import annotations

from typing import Dict, List
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import THEME_KEYWORDS
from .utils import normalize_text


def extract_themes(comments: pd.DataFrame) -> pd.DataFrame:
    theme_counts = {name: [] for name in THEME_KEYWORDS}
    for _, row in comments.iterrows():
        text = normalize_text(str(row.get('all_text', '')))
        for name, keywords in THEME_KEYWORDS.items():
            count = sum(text.count(k) for k in keywords)
            theme_counts[name].append(count)
    theme_df = pd.DataFrame(theme_counts)
    theme_df['row_index'] = comments['row_index'].values
    return theme_df


def compare_comments_by_condition(comments: pd.DataFrame) -> pd.DataFrame:
    data = comments[['condition', 'all_text']].fillna("")
    results = []
    for cond, sub in data.groupby('condition'):
        texts = sub['all_text'].tolist()
        if not texts:
            continue
        vec = TfidfVectorizer(max_features=50, stop_words='english')
        try:
            X = vec.fit_transform(texts)
        except ValueError:
            continue
        terms = vec.get_feature_names_out()
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:15]
        for i in top_idx:
            results.append({"condition": cond, "term": terms[i], "avg_tfidf": float(scores[i])})
    return pd.DataFrame(results)


def map_justifications_by_anchor_and_task(comments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in comments.iterrows():
        for t in ['key', 'visual', 'controls']:
            text = normalize_text(str(r.get(f'comment_{t}', '')))
            if not text:
                continue
            rows.append({
                "participant_id": r.get('participant_id'),
                "condition": r.get('condition'),
                "task": t.capitalize(),
                "mentions_world": int('world' in text),
                "mentions_head": int('head' in text),
                "mentions_torso": int('torso' in text or 'body' in text),
                "mentions_arm": int('arm' in text or 'wrist' in text or 'hand' in text),
            })
    return pd.DataFrame(rows)


def select_representative_quotes(comments: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    reps = []
    def theme_score(text: str, keywords: List[str]) -> int:
        ntext = normalize_text(text)
        return sum(ntext.count(k) for k in keywords)
    for theme, keywords in THEME_KEYWORDS.items():
        scored = []
        for _, r in comments.iterrows():
            text = str(r.get('all_text', ''))
            if not text.strip():
                continue
            score = theme_score(text, keywords)
            if score > 0:
                sent = analyzer.polarity_scores(text)
                magnitude = abs(sent['compound'])
                scored.append((magnitude * score, text, r.get('condition')))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s in scored[:3]:
            reps.append({"theme": theme, "quote": s[1], "condition": s[2], "score": s[0]})
    return pd.DataFrame(reps)


# Transformer-based sentiment (state-of-the-art multilingual model)

_SENTIMENT_PIPELINE = None

def _get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None and _HAS_TRANSFORMERS:
        # cardiffnlp/twitter-xlm-roberta-base-sentiment is a strong multilingual sentiment model
        try:
            _SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", truncation=True)
        except Exception:
            _SENTIMENT_PIPELINE = None
    return _SENTIMENT_PIPELINE


def compute_transformer_sentiment(comments: pd.DataFrame) -> pd.DataFrame:
    texts = comments[['row_index', 'participant_id', 'condition', 'all_text']].copy()
    texts['all_text'] = texts['all_text'].fillna("").astype(str)
    out_rows = []
    pipe = _get_sentiment_pipeline()
    if pipe is None:
        # Fallback to VADER if transformers unavailable
        analyzer = SentimentIntensityAnalyzer()
        for _, r in texts.iterrows():
            s = analyzer.polarity_scores(r['all_text'])
            out_rows.append({
                'row_index': r['row_index'],
                'participant_id': r['participant_id'],
                'condition': r['condition'],
                'label': 'POS' if s['compound'] > 0.05 else ('NEG' if s['compound'] < -0.05 else 'NEU'),
                'score': float(abs(s['compound'])),
                'compound': float(s['compound']),
            })
    else:
        # Use transformer pipeline
        for _, r in texts.iterrows():
            text = r['all_text'][:512]  # keep reasonably short for inference
            res = pipe(text)[0]
            out_rows.append({
                'row_index': r['row_index'],
                'participant_id': r['participant_id'],
                'condition': r['condition'],
                'label': res.get('label'),
                'score': float(res.get('score', 0.0)),
            })
    return pd.DataFrame(out_rows)


def aggregate_sentiment(sent_df: pd.DataFrame, comments: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Add task-specific comment fields by melting individual comment columns
    task_rows = []
    for _, r in comments.iterrows():
        for t in ['Key', 'Visual', 'Controls']:
            col = f"comment_{t.lower()}"
            txt = str(r.get(col, ''))
            if txt and txt.strip():
                task_rows.append({
                    'row_index': r['row_index'],
                    'participant_id': r['participant_id'],
                    'condition': r['condition'],
                    'task': t,
                    'text': txt,
                })
    tasks_df = pd.DataFrame(task_rows)

    # Merge sentence-level labels back to rows
    if 'label' in sent_df.columns:
        label_map = sent_df.groupby('row_index')['label'].agg(lambda s: s.iloc[0]).rename('overall_label')
        sent_merged = tasks_df.merge(label_map, left_on='row_index', right_index=True, how='left')
    else:
        sent_merged = tasks_df.copy()

    # Condition x task distribution of labels
    if 'overall_label' in sent_merged.columns:
        dist = (sent_merged.groupby(['condition', 'task', 'overall_label'])
                .size().reset_index(name='count'))
        dist['proportion'] = dist.groupby(['condition', 'task'])['count'].transform(lambda s: s / s.sum())
    else:
        dist = pd.DataFrame()

    return {
        'by_condition_task': dist,
        'task_texts': tasks_df,
    }
