try:
    import transformer_engine.pytorch as te  # type: ignore
    TEDotProductAttention = te.DotProductAttention
except (ImportError, AttributeError):
    te = None
    TEDotProductAttention = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None and TEDotProductAttention is not None
