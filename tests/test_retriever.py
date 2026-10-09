"""Test cases for the RetrieverAdapter module in aap_dspy."""

import pytest
from aap_core.types import AgentMessage
from aap_dspy.retriever import RetrieverAdapter

import dspy


class _MockRetrieve(dspy.Retrieve):
    """Custom dspy.Retrieve subclass that returns mock passages."""

    def __init__(self, passages: list[str], **kwargs):
        super().__init__(**kwargs)
        self._passages = passages

    def forward(self, query: str, k: int | None = None, **kwargs):
        return self._passages


class _MockEmbeddings(dspy.Embeddings):
    """Custom dspy.Embeddings subclass that returns mock passages."""

    def __init__(self, passages: list[str], **kwargs):
        # Provide a minimal mock embedder function
        mock_embedder = lambda x: [[0.0] * 10 for _ in x]
        super().__init__(corpus=passages, embedder=mock_embedder, **kwargs)
        self._passages = passages

    def forward(self, query: str, **kwargs):
        # Return an object with passages attribute like real Embeddings does
        return type("MockResult", (), {"passages": self._passages})()


class TestRetrieverAdapterDataKeyValidation:
    """Test data_key field validation."""

    def test_valid_data_key_default(self):
        """Test creating a retriever with default data_key."""
        retriever = _MockRetrieve(passages=[])
        adapter = RetrieverAdapter(retriever=retriever)
        assert adapter.data_key == "context.data"

    def test_valid_data_key_custom(self):
        """Test creating a retriever with a custom valid data_key."""
        retriever = _MockRetrieve(passages=[])
        adapter = RetrieverAdapter(retriever=retriever, data_key="context.custom_field")
        assert adapter.data_key == "context.custom_field"

    def test_valid_data_key_nested(self):
        """Test creating a retriever with a nested custom data_key."""
        retriever = _MockRetrieve(passages=[])
        adapter = RetrieverAdapter(
            retriever=retriever, data_key="context.retrieval.docs"
        )
        assert adapter.data_key == "context.retrieval.docs"

    def test_invalid_data_key_no_prefix(self):
        """Test that data_key without 'context.' prefix raises ValueError."""
        retriever = _MockRetrieve(passages=[])
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=retriever, data_key="data")

    def test_invalid_data_key_wrong_prefix(self):
        """Test that data_key with wrong prefix raises ValueError."""
        retriever = _MockRetrieve(passages=[])
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=retriever, data_key="message.data")

    def test_invalid_data_key_empty(self):
        """Test that empty data_key raises ValueError."""
        retriever = _MockRetrieve(passages=[])
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=retriever, data_key="")


class TestRetrieverAdapterRetrieveWithRetrieve:
    """Test retrieve method using real dspy.Retrieve subclass."""

    def test_retrieve_no_existing_context(self):
        """Test retrieve when message has no existing context."""
        passages = ["Bird species A is endangered.", "Bird species B is vulnerable."]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="What birds are endangered?")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["data"] == (
            "Bird species A is endangered. Bird species B is vulnerable."
        )

    def test_retrieve_with_existing_context(self):
        """Test retrieve when message already has context."""
        passages = ["New bird info"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(
            query="What birds are endangered?",
            context={"other_key": "existing_value"},
        )

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["other_key"] == "existing_value"
        assert result.context["data"] == ["New bird info"]

    def test_retrieve_with_custom_data_key(self):
        """Test retrieve with a custom data_key."""
        passages = ["Custom bird data"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever, data_key="context.bird_info")
        message = AgentMessage(query="Test query")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["bird_info"] == ["Custom bird data"]

    def test_retrieve_with_existing_context_same_key(self):
        """Test retrieve overwrites existing data at the same data_key."""
        passages = ["Updated bird data"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Test query", context={"data": ["old content"]})

        result = adapter.retrieve(message)

        assert result.context["data"] == ["Updated bird data"]

    def test_retrieve_single_passage(self):
        """Test retrieve with a single passage."""
        passages = ["Single bird fact"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Single fact query")

        result = adapter.retrieve(message)

        assert result.context["data"] == ["Single bird fact"]

    def test_retrieve_empty_results(self):
        """Test retrieve when retriever returns no passages."""
        passages = []
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="No results query")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["data"] == []

    def test_retrieve_multiple_passages(self):
        """Test retrieve with multiple passages."""
        passages = [
            "Eagle - apex predator",
            "Penguin - flightless bird",
            "Ostrich - largest bird",
            "Hummingbird - smallest bird",
        ]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Multi passage query")

        result = adapter.retrieve(message)

        assert result.context["data"] == (
            "Eagle - apex predator Penguin - flightless bird "
            "Ostrich - largest bird Hummingbird - smallest bird"
        )

    def test_retrieve_preserves_message_query(self):
        """Test that retrieve preserves the original query."""
        passages = ["Doc"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Original query")

        result = adapter.retrieve(message)

        assert result.query == "Original query"

    def test_retrieve_preserves_other_message_fields(self):
        """Test that retrieve preserves other message fields."""
        passages = ["Doc"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(
            query="Test query",
            origin="test_agent",
            responses=[("agent1", "response 1")],
        )

        result = adapter.retrieve(message)

        assert result.origin == "test_agent"
        assert result.responses == [("agent1", "response 1")]

    def test_retrieve_with_kwargs_passed_to_retriever(self):
        """Test that kwargs are passed through to the retriever."""
        passages = ["Doc"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Test query")

        # Should not raise - kwargs are accepted but not used by our mock
        result = adapter.retrieve(message, top_k=5, filter_by=["category:birds"])

        assert result.context is not None


class TestRetrieverAdapterRetrieveWithEmbeddings:
    """Test retrieve method using real dspy.Embeddings subclass."""

    def test_retrieve_with_embeddings_no_existing_context(self):
        """Test retrieve using Embeddings when message has no existing context."""
        passages = ["Mammal A is endangered.", "Mammal B is vulnerable."]
        embeddings = _MockEmbeddings(passages=passages)
        adapter = RetrieverAdapter(retriever=embeddings)
        message = AgentMessage(query="What mammals are endangered?")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["data"] == (
            "Mammal A is endangered. Mammal B is vulnerable."
        )

    def test_retrieve_with_embeddings_single_passage(self):
        """Test retrieve using Embeddings with a single passage."""
        passages = ["Mammal fact"]
        embeddings = _MockEmbeddings(passages=passages)
        adapter = RetrieverAdapter(retriever=embeddings)
        message = AgentMessage(query="Single fact query")

        result = adapter.retrieve(message)

        assert result.context["data"] == ["Mammal fact"]

    def test_retrieve_with_embeddings_empty_results(self):
        """Test retrieve using Embeddings when no passages are returned."""
        # Use a non-empty corpus but return empty passages from forward
        passages = ["placeholder"]
        embeddings = _MockEmbeddings(passages=passages)
        # Override forward to return empty passages
        original_forward = embeddings.forward
        embeddings.forward = lambda query, **kwargs: type(
            "MockResult", (), {"passages": []}
        )()
        adapter = RetrieverAdapter(retriever=embeddings)
        message = AgentMessage(query="No results query")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["data"] == []

    def test_retrieve_with_embeddings_custom_data_key(self):
        """Test retrieve using Embeddings with a custom data_key."""
        passages = ["Custom mammal data"]
        embeddings = _MockEmbeddings(passages=passages)
        adapter = RetrieverAdapter(retriever=embeddings, data_key="context.mammal_info")
        message = AgentMessage(query="Test query")

        result = adapter.retrieve(message)

        assert result.context is not None
        assert result.context["mammal_info"] == ["Custom mammal data"]

    def test_retrieve_with_embeddings_preserves_query(self):
        """Test that retrieve using Embeddings preserves the original query."""
        passages = ["Doc"]
        embeddings = _MockEmbeddings(passages=passages)
        adapter = RetrieverAdapter(retriever=embeddings)
        message = AgentMessage(query="Original query")

        result = adapter.retrieve(message)

        assert result.query == "Original query"


class TestRetrieverAdapterEdgeCases:
    """Test edge cases for RetrieverAdapter."""

    def test_retrieve_with_many_passages(self):
        """Test retrieve with a large number of passages."""
        passages = [f"Passage {i} content" for i in range(100)]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Large retrieval query")

        result = adapter.retrieve(message)

        assert result.context is not None
        expected = " ".join(passages)
        assert result.context["data"] == expected

    def test_retrieve_with_empty_string_passage(self):
        """Test retrieve when a passage is an empty string."""
        passages = ["", "non-empty passage"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Empty passage query")

        result = adapter.retrieve(message)

        assert result.context["data"] == " non-empty passage"

    def test_retrieve_with_whitespace_only_passage(self):
        """Test retrieve when a passage is whitespace only."""
        passages = ["   ", "valid passage"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Whitespace passage query")

        result = adapter.retrieve(message)

        assert result.context["data"] == "    valid passage"

    def test_retrieve_with_special_characters(self):
        """Test retrieve with passages containing special characters."""
        passages = [
            "Passage with <html> tags",
            "Passage with 'quotes' and \"double quotes\"",
        ]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Special chars query")

        result = adapter.retrieve(message)

        assert result.context["data"] == (
            "Passage with <html> tags Passage with 'quotes' and \"double quotes\""
        )

    def test_retrieve_with_unicode_characters(self):
        """Test retrieve with passages containing unicode characters."""
        passages = ["Bird: 鳥", "Oiseau: 🐦", "Ave: 🦅"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Unicode query")

        result = adapter.retrieve(message)

        assert result.context["data"] == "Bird: 鳥 Oiseau: 🐦 Ave: 🦅"

    def test_retrieve_overwrites_existing_context_value(self):
        """Test that retrieve overwrites existing context value at the same key."""
        passages = ["New data"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(
            query="Test query",
            context={"data": ["old data 1", "old data 2"]},
        )

        result = adapter.retrieve(message)

        assert result.context["data"] == ["New data"]

    def test_retrieve_with_newline_in_passage(self):
        """Test retrieve with passages containing newline characters."""
        passages = ["Line 1\nLine 2", "Line 3\nLine 4"]
        retriever = _MockRetrieve(passages=passages)
        adapter = RetrieverAdapter(retriever=retriever)
        message = AgentMessage(query="Newline query")

        result = adapter.retrieve(message)

        assert result.context["data"] == "Line 1\nLine 2 Line 3\nLine 4"
