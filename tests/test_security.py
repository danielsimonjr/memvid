"""
Security hardening tests for HuggingFace model loading.

Covers the supply-chain RCE mitigations introduced for IndexManager:
  - SentenceTransformer is loaded with trust_remote_code=False
  - The default model name is fully qualified (org/name) and pinned to a revision
  - MEMVID_EMBEDDING_MODEL and MEMVID_EMBEDDING_REVISION env vars override defaults
  - TRANSFORMERS_USE_SAFETENSORS=1 is set at import time (preference for safetensors weights)

These tests use mocks so they do NOT touch the network or download model weights.
"""

import os
import importlib
from unittest.mock import patch, MagicMock

import pytest


def _reload_index_module():
    """Reimport memvid.index so module-level code (env vars) re-runs."""
    import memvid.index
    return importlib.reload(memvid.index)


def test_safetensors_env_var_set_on_import():
    """Module import must set TRANSFORMERS_USE_SAFETENSORS=1 to prefer safetensors weights."""
    _reload_index_module()
    assert os.environ.get("TRANSFORMERS_USE_SAFETENSORS") == "1"


def test_default_model_is_pinned_and_qualified():
    """Default config must use the org-qualified model name and a pinned revision SHA."""
    from memvid.config import EMBEDDING_MODEL, EMBEDDING_REVISION
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2", (
        "Default embedding model must be fully qualified to prevent name-squatting"
    )
    # Revision must be a 40-char hex SHA (HuggingFace commit hash)
    assert isinstance(EMBEDDING_REVISION, str)
    assert len(EMBEDDING_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in EMBEDDING_REVISION), (
        "EMBEDDING_REVISION must be a full 40-char commit SHA, not a branch/tag"
    )


def test_index_manager_passes_trust_remote_code_false():
    """IndexManager must pass trust_remote_code=False to SentenceTransformer."""
    index_mod = _reload_index_module()
    with patch.object(index_mod, "SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        index_mod.IndexManager()
        assert mock_st.called, "SentenceTransformer was not called"
        _, kwargs = mock_st.call_args
        assert kwargs.get("trust_remote_code") is False, (
            f"trust_remote_code must be False, got {kwargs.get('trust_remote_code')!r}"
        )


def test_index_manager_passes_pinned_revision():
    """IndexManager must pass a pinned revision SHA to SentenceTransformer."""
    index_mod = _reload_index_module()
    from memvid.config import EMBEDDING_REVISION
    with patch.object(index_mod, "SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        index_mod.IndexManager()
        _, kwargs = mock_st.call_args
        assert kwargs.get("revision") == EMBEDDING_REVISION


def test_env_var_overrides_default_model():
    """MEMVID_EMBEDDING_MODEL env var must override the default model name."""
    override = "sentence-transformers/all-mpnet-base-v2"
    with patch.dict(os.environ, {"MEMVID_EMBEDDING_MODEL": override}, clear=False):
        index_mod = _reload_index_module()
        with patch.object(index_mod, "SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            index_mod.IndexManager()
            args, kwargs = mock_st.call_args
            # First positional arg is the model name
            assert args[0] == override, f"expected {override}, got {args[0]!r}"


def test_env_var_overrides_default_revision():
    """MEMVID_EMBEDDING_REVISION env var must override the pinned revision."""
    override = "deadbeefcafebabe1234567890abcdef12345678"
    with patch.dict(os.environ, {"MEMVID_EMBEDDING_REVISION": override}, clear=False):
        index_mod = _reload_index_module()
        with patch.object(index_mod, "SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            index_mod.IndexManager()
            _, kwargs = mock_st.call_args
            assert kwargs.get("revision") == override


def test_explicit_config_model_still_honored():
    """Caller-provided config['embedding']['model'] must take precedence over env defaults."""
    index_mod = _reload_index_module()
    custom_cfg = {
        "embedding": {
            "model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "revision": "abcdef0123456789abcdef0123456789abcdef01",
            "dimension": 384,
        },
        "index": {"type": "Flat", "nlist": 100},
    }
    with patch.object(index_mod, "SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        index_mod.IndexManager(config=custom_cfg)
        args, kwargs = mock_st.call_args
        assert args[0] == "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        assert kwargs.get("revision") == "abcdef0123456789abcdef0123456789abcdef01"
        assert kwargs.get("trust_remote_code") is False
