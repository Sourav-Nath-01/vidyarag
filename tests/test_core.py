"""
tests/test_core.py  —  Unit tests for core retrieval components
===============================================================
Tests cover the deterministic, side-effect-free components of the pipeline:
  - BM25 tokeniser
  - Reciprocal Rank Fusion
  - Relevance matching (_is_correct)
  - JSON extractor for LLM output
  - Intent heuristic classifier
  - Passage text builder

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── path setup ─────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
for _candidate in [_root / "src" / "retrieval"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break


# =============================================================================
# 1. BM25 Tokeniser
# =============================================================================

class TestBM25Tokeniser:
    """Tests for the _tokenise function in retriever.py"""

    def test_basic_tokenisation(self):
        from retriever import _tokenise
        tokens = _tokenise("binary search tree insertion")
        assert "binary" in tokens
        assert "search" in tokens
        assert "tree" in tokens
        assert "insertion" in tokens

    def test_stopwords_removed(self):
        from retriever import _tokenise
        # "the", "a", "is", "it" are all in _STOPWORDS
        tokens = _tokenise("the binary tree is a data structure")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "is" not in tokens
        assert "binary" in tokens
        assert "data" in tokens

    def test_case_insensitive(self):
        from retriever import _tokenise
        tokens_lower = _tokenise("backpropagation gradient descent")
        tokens_upper = _tokenise("Backpropagation Gradient Descent")
        assert tokens_lower == tokens_upper

    def test_short_tokens_removed(self):
        from retriever import _tokenise
        # Single-character tokens should be removed (len >= 2 required)
        tokens = _tokenise("a b c d BST")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "bst" in tokens   # lowercased

    def test_punctuation_stripped(self):
        from retriever import _tokenise
        tokens = _tokenise("what is O(n log n) complexity?")
        # Parens and ? should be stripped
        assert "(" not in " ".join(tokens)
        assert "?" not in " ".join(tokens)

    def test_empty_string(self):
        from retriever import _tokenise
        assert _tokenise("") == []

    def test_returns_list_of_strings(self):
        from retriever import _tokenise
        result = _tokenise("deep learning neural networks")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)


# =============================================================================
# 2. Reciprocal Rank Fusion
# =============================================================================

class TestRRF:
    """Tests for _reciprocal_rank_fusion in retriever.py"""

    def test_basic_fusion(self):
        from retriever import _reciprocal_rank_fusion
        dense  = [(0, 0.9), (1, 0.8), (2, 0.7)]
        sparse = [(1, 5.0), (2, 4.0), (3, 3.0)]
        result = _reciprocal_rank_fusion(dense, sparse)
        # Results should be sorted descending by RRF score
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_union_of_indices(self):
        from retriever import _reciprocal_rank_fusion
        dense  = [(0, 0.9), (1, 0.8)]
        sparse = [(2, 5.0), (3, 4.0)]
        result = _reciprocal_rank_fusion(dense, sparse)
        result_indices = {idx for idx, _ in result}
        assert result_indices == {0, 1, 2, 3}

    def test_overlap_boosted(self):
        """An index appearing in both lists should score higher than one appearing in only one."""
        from retriever import _reciprocal_rank_fusion
        dense  = [(0, 0.95), (1, 0.5)]
        sparse = [(0, 10.0), (1, 0.1)]
        result = _reciprocal_rank_fusion(dense, sparse)
        # idx 0 appears in both at rank 1 in both — should be top
        top_idx = result[0][0]
        assert top_idx == 0

    def test_empty_dense(self):
        from retriever import _reciprocal_rank_fusion
        dense  = []
        sparse = [(0, 5.0), (1, 4.0)]
        result = _reciprocal_rank_fusion(dense, sparse)
        assert len(result) == 2

    def test_empty_sparse(self):
        from retriever import _reciprocal_rank_fusion
        dense  = [(0, 0.9), (1, 0.8)]
        sparse = []
        result = _reciprocal_rank_fusion(dense, sparse)
        assert len(result) == 2

    def test_both_empty(self):
        from retriever import _reciprocal_rank_fusion
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_scores_are_positive(self):
        from retriever import _reciprocal_rank_fusion
        dense  = [(i, float(10 - i)) for i in range(5)]
        sparse = [(i, float(5 - i)) for i in range(3)]
        result = _reciprocal_rank_fusion(dense, sparse)
        assert all(score > 0 for _, score in result)

    def test_custom_k_rrf(self):
        """Higher k_rrf → smaller score differences between ranks."""
        from retriever import _reciprocal_rank_fusion
        dense  = [(0, 0.9), (1, 0.8)]
        sparse = []
        result_k1  = _reciprocal_rank_fusion(dense, sparse, k_rrf=1)
        result_k60 = _reciprocal_rank_fusion(dense, sparse, k_rrf=60)
        # Higher k_rrf → smaller absolute scores
        assert result_k1[0][1] > result_k60[0][1]


# =============================================================================
# 3. Relevance Matching (_is_correct)
# =============================================================================

class TestIsCorrect:
    """Tests for _is_correct in evaluator.py"""

    def _get_is_correct(self):
        _eval_root = _root / "src" / "retrieval"
        sys.path.insert(0, str(_eval_root))
        # evaluator.py tries to load annotations at import time — mock that
        import unittest.mock as mock
        import builtins
        # Patch open so load_annotations doesn't fail at import
        fake_line = json.dumps({
            "id": "q1", "query": "test", "status": "selected",
            "expected_course": "dsa", "expected_start_sec": 60,
        })
        with mock.patch("builtins.open", mock.mock_open(read_data=fake_line)):
            import importlib
            if "evaluator" in sys.modules:
                del sys.modules["evaluator"]
            import evaluator
        return evaluator._is_correct

    def test_correct_course_and_timestamp(self):
        _is_correct = self._get_is_correct()
        result = {
            "course_name": "Introduction to Algorithms and Analysis",
            "start_sec": 120.0,
        }
        query_meta = {
            "expected_course": "dsa",
            "expected_start_sec": 150,
        }
        # 150 - 120 = 30s — within 90s tolerance
        assert _is_correct(result, query_meta) is True

    def test_wrong_course(self):
        _is_correct = self._get_is_correct()
        result = {
            "course_name": "Deep Learning",
            "start_sec": 120.0,
        }
        query_meta = {
            "expected_course": "dsa",
            "expected_start_sec": 120,
        }
        assert _is_correct(result, query_meta) is False

    def test_correct_course_wrong_timestamp(self):
        _is_correct = self._get_is_correct()
        result = {
            "course_name": "Introduction to Algorithms and Analysis",
            "start_sec": 500.0,
        }
        query_meta = {
            "expected_course": "dsa",
            "expected_start_sec": 100,   # 400s apart — exceeds 90s tolerance
        }
        assert _is_correct(result, query_meta) is False

    def test_no_timestamp_annotation(self):
        """When expected_start_sec=0, only course match is checked."""
        _is_correct = self._get_is_correct()
        result = {
            "course_name": "Introduction to Algorithms and Analysis",
            "start_sec": 999.0,
        }
        query_meta = {
            "expected_course": "dsa",
            "expected_start_sec": 0,
        }
        assert _is_correct(result, query_meta) is True

    def test_none_course_name(self):
        _is_correct = self._get_is_correct()
        result = {"course_name": None, "start_sec": 60.0}
        query_meta = {"expected_course": "dsa", "expected_start_sec": 60}
        assert _is_correct(result, query_meta) is False

    def test_exact_boundary_tolerance(self):
        """start_sec exactly at 90s boundary should still be correct."""
        _is_correct = self._get_is_correct()
        result = {
            "course_name": "Introduction to Algorithms and Analysis",
            "start_sec": 200.0,
        }
        query_meta = {
            "expected_course": "dsa",
            "expected_start_sec": 290,   # exactly 90s gap
        }
        assert _is_correct(result, query_meta) is True


# =============================================================================
# 4. JSON Extractor (LLM output parsing)
# =============================================================================

class TestJSONExtractor:
    """Tests for _extract_json_object in retriever.py"""

    def test_clean_json(self):
        from retriever import _extract_json_object
        text = '{"intent": "code", "expanded_query": "BST", "reasoning": "ok"}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["intent"] == "code"

    def test_json_with_preamble(self):
        from retriever import _extract_json_object
        text = 'Here is my analysis:\n{"intent": "theoretical", "expanded_query": "merge sort", "reasoning": "t"}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["intent"] == "theoretical"

    def test_markdown_fenced(self):
        from retriever import _extract_json_object
        text = '```json\n{"intent": "conceptual", "expanded_query": "VM", "reasoning": "c"}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert result["intent"] == "conceptual"

    def test_braces_inside_string(self):
        from retriever import _extract_json_object
        text = '{"intent": "code", "expanded_query": "BST {tree}", "reasoning": "has braces"}'
        result = _extract_json_object(text)
        assert result is not None
        assert result["intent"] == "code"

    def test_no_json(self):
        from retriever import _extract_json_object
        result = _extract_json_object("Sorry, I cannot answer that.")
        assert result is None

    def test_trailing_text(self):
        from retriever import _extract_json_object
        text = '{"intent": "code"} some trailing text here'
        result = _extract_json_object(text)
        assert result is not None
        assert result["intent"] == "code"

    def test_empty_string(self):
        from retriever import _extract_json_object
        result = _extract_json_object("")
        assert result is None


# =============================================================================
# 5. Intent Heuristic Classifier
# =============================================================================

class TestIntentHeuristic:
    """Tests for _detect_intent_heuristic in retriever.py"""

    def test_code_intent(self):
        from retriever import _detect_intent_heuristic
        assert _detect_intent_heuristic("write python code for bubble sort") == "code"
        assert _detect_intent_heuristic("implement a stack class in python") == "code"

    def test_theoretical_intent(self):
        from retriever import _detect_intent_heuristic
        assert _detect_intent_heuristic("explain the complexity of merge sort") == "theoretical"
        assert _detect_intent_heuristic("what is the definition of a binary tree") == "theoretical"

    def test_conceptual_default(self):
        from retriever import _detect_intent_heuristic
        # Generic query with no strong signals → conceptual
        result = _detect_intent_heuristic("how does virtual memory work")
        assert result in ("conceptual", "theoretical")  # "how" and "does" — ambiguous, acceptable

    def test_empty_query(self):
        from retriever import _detect_intent_heuristic
        # Should not raise
        result = _detect_intent_heuristic("")
        assert result == "conceptual"


# =============================================================================
# 6. Passage Text Builder (embedder)
# =============================================================================

class TestPassageBuilder:
    """Tests for build_passage_text in embedder.py"""

    def _get_builder(self):
        sys.path.insert(0, str(_root / "src" / "retrieval"))
        from embedder import build_passage_text
        return build_passage_text

    def test_passage_prefix_present(self):
        build = self._get_builder()
        seg = {
            "course_name": "Deep Learning",
            "lecture_title": "Backpropagation",
            "content_type": "theoretical",
            "transcript": "Backpropagation computes gradients.",
            "ocr_text": "",
            "ocr_failed": True,
        }
        text = build(seg)
        assert text.startswith("passage:")

    def test_course_and_lecture_in_text(self):
        build = self._get_builder()
        seg = {
            "course_name": "Operating Systems",
            "lecture_title": "Virtual Memory",
            "content_type": "conceptual",
            "transcript": "Virtual memory extends RAM.",
            "ocr_text": "",
            "ocr_failed": True,
        }
        text = build(seg)
        assert "Operating Systems" in text
        assert "Virtual Memory" in text

    def test_ocr_included_when_available(self):
        build = self._get_builder()
        seg = {
            "course_name": "Algorithms",
            "lecture_title": "BST",
            "content_type": "conceptual",
            "transcript": "Let us insert a node.",
            "ocr_text": "Binary Search Tree Insert Operation",
            "ocr_failed": False,
        }
        text = build(seg)
        assert "Binary Search Tree" in text
        assert "[SLIDE]" in text

    def test_failed_ocr_excluded(self):
        build = self._get_builder()
        seg = {
            "course_name": "Algorithms",
            "lecture_title": "BST",
            "content_type": "conceptual",
            "transcript": "Let us insert a node.",
            "ocr_text": "some slide text",
            "ocr_failed": True,   # OCR failed — should be excluded
        }
        text = build(seg)
        assert "[SLIDE]" not in text

    def test_trivially_short_ocr_excluded(self):
        build = self._get_builder()
        seg = {
            "course_name": "ML",
            "lecture_title": "SVM",
            "content_type": "theoretical",
            "transcript": "Support vector machines separate classes.",
            "ocr_text": "ok",   # len <= 10 → excluded
            "ocr_failed": False,
        }
        text = build(seg)
        assert "[SLIDE]" not in text

    def test_empty_segment(self):
        build = self._get_builder()
        seg = {}
        text = build(seg)
        # Should not raise and should still have passage prefix
        assert "passage:" in text
