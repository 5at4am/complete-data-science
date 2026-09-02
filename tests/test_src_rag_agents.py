"""Tests for src.rag.chunker, src.rag.retriever, and src.agents.tool."""

from src.agents.tool import execute_tool_call, tool_schema
from src.rag.chunker import chunk_documents, chunk_text
from src.rag.retriever import TfidfRetriever


def test_chunk_text_splits_long_text():
    text = "Sentence one. Sentence two. " * 200  # will not fit in one chunk
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    # each chunk = at most chunk_size chars + the overlap prefix + breathing room
    assert all(len(c) <= 620 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("Hello world. This is short.", chunk_size=500, overlap=50)
    assert chunks == ["Hello world. This is short."]


def test_chunk_text_overlap_carries_context():
    text = "Alpha content. " * 60 + "KEY TERM lives here. " + "Beta content. " * 60
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert any("KEY TERM" in c for c in chunks)
    assert len(chunks) > 1
    # overlapping chunks preserve the boundary
    joined = " ".join(chunks)
    assert "Alpha content." in joined


def test_chunk_text_invalid_args():
    try:
        chunk_text("abc", chunk_size=100, overlap=200)
        assert False, "should have raised"
    except ValueError:
        pass


def test_chunk_documents_flat():
    chunks = chunk_documents(["Doc one. Doc two.", "Doc three."], chunk_size=200, overlap=10)
    assert len(chunks) == 2


def test_tfidf_retriever_returns_relevant_chunk():
    corpus = [
        "The cat sat on the mat.",
        "Gradient boosting builds trees sequentially.",
        "SHAP values explain model predictions.",
    ]
    retriever = TfidfRetriever().index(corpus)
    top = retriever.search("explain predictions", k=1)
    assert top[0][0] == 2  # SHAP chunk is most relevant
    assert 0.0 < top[0][1] <= 1.0


def test_tool_schema_builds_openai_shape():
    schema = tool_schema(
        "add", "Adds two numbers", {"a": {"type": int, "description": "first"}, "b": {"type": int, "description": "second"}}
    )
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert "a" in schema["function"]["parameters"]["properties"]


def test_execute_tool_call_dispatch_and_error():
    tools = {"double": lambda x: x * 2}
    assert execute_tool_call(tools, "double", {"x": 21}) == "42"
    assert execute_tool_call(tools, "nope", {}) == "ERROR: unknown tool 'nope'"
    bad = execute_tool_call({"recip": lambda x: 1 / x}, "recip", {"x": 0})
    assert bad.startswith("ERROR:")
    assert "ZeroDivisionError" in bad  # runtime errors are surfaced as text