"""Contract tests for the TOON prettier (aap_core 0.3.1) through the DSPy adapter.

The TOON encoder itself lives in ``aap_core``: ``DataFrameRetriever.from_pandas``
and ``DataFrameRetriever.from_dict`` accept ``prettier="toon"`` and encode through
``toon_format`` (declared by aap_core as ``toon-format>=1.0.0``). What this module
pins down is the boundary ``aap_dspy`` owns, namely that a TOON payload survives
the trip

    retriever -> AgentMessage.context -> BaseSignatureAdapter.msg2sig
              -> dspy.Signature -> ChatAdapter -> dspy.LM.__call__(messages=...)

verbatim, at the layer the model actually receives.

Two load-bearing risks drive the assertions:

1. TOON's tabular header is brace-bearing (``"[2]{sku,qty}:"``). DSPy renders the
   field structure into the *system* message as literal ``{context}`` placeholders
   and interpolates values into the *user* message with f-strings
   (``dspy/adapters/chat_adapter.py``: ``f"[[ ## {k} ## ]]\\n{value}"``). Values are
   never re-scanned, so the payload must arrive byte-for-byte; and
   ``ChatCausalMultiTurnsChain._generate_response`` round-trips the signature
   through ``model_dump()``/reconstruction, which must not re-quote it either.
2. ``RetrieverAdapter.retrieve`` used to store
   ``" ".join(data) if len(data) > 1 else data`` - a single hit became a bare
   ``list`` (a hard validation error for a ``str``-annotated ``dspy.InputField``)
   and multiple hits a space-joined ``str`` that glued prose onto the last TOON
   row. It now stores ``"\n".join(...)`` unconditionally, so the stored *type* no
   longer depends on hit count. What remains open is that TOON is a single-root
   document format: a table plus a prose passage in one value is not decodable with
   any separator. That gap is pinned with strict xfail, never weakened.

Offline by design: the LM is the repo's ``MockLM`` pattern from
``tests/test_chain.py`` (a ``dspy.LM`` subclass whose ``__call__`` only records the
messages dspy built). No network, no provider call, no real LM.
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List

import dspy
import pandas as pd
import pytest
import toon_format
from aap_core.retriever import DataFrameRetriever
from aap_core.types import AgentMessage
from pydantic import ValidationError

import aap_dspy.chain as chain_module
from aap_dspy.chain import BaseSignatureAdapter, ChatCausalMultiTurnsChain
from aap_dspy.retriever import RetrieverAdapter

# The aap_core release that routed prettier="toon" to toon_format.
TOON_MIN_AAP_CORE = (0, 3, 1)
# Where dspy puts a field's value in the user message, and where the next field
# starts. Taken from dspy's ChatAdapter layout, not invented here.
CONTEXT_MARKER = "[[ ## context ## ]]\n"
QUESTION_MARKER = "\n\n[[ ## question ## ]]\n"
TABLE = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
PAYLOAD = toon_format.encode(TABLE)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep capability detection inert so no test can reach the Hugging Face Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


class MockLM(dspy.LM):
    """LM stub that records adapter output instead of calling a provider.

    Same shape as ``tests/test_chain.py``: ``__call__`` returns the raw completion
    text dspy would parse, so nothing below it knows it is not online.
    """

    def __init__(
        self, model: str = "openai/gpt-4o-mini", reply: str = "It is a table."
    ):
        super().__init__(model, cache=False, max_tokens=16)
        self.reply = reply
        self.captured: List[dict] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.captured.append({"messages": messages, "kwargs": kwargs})
        return [f"[[ ## answer ## ]]\n{self.reply}\n[[ ## completed ## ]]"]


class ToonSig(dspy.Signature):
    """Answer the question using the context table."""

    context: str = dspy.InputField(desc="a TOON-encoded table")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ToonListSig(dspy.Signature):
    """Same contract, but the context field is typed as a list of strings."""

    context: List[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ToonHistorySig(dspy.Signature):
    """A context table plus replayed history, the dspy multi-turn shape."""

    history: dspy.History = dspy.InputField(default=None)
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class TableSig(dspy.Signature):
    """A table supplied as static prefill rather than per-message context."""

    table: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ContextAdapter(BaseSignatureAdapter):
    """Maps ``AgentMessage.context[data_key]`` onto the signature's context field."""

    def __init__(self, field: str = "context", data_key: str = "data"):
        self.field = field
        self.data_key = data_key

    def msg2sig(self, message: AgentMessage) -> List[Any]:
        value = (message.context or {}).get(self.data_key, "")
        sig = ToonListSig if self.field == "list_context" else ToonSig
        name = "context"
        return [sig(**{name: value, "question": message.query, "answer": ""})]

    def sig2msg(self, signatures: List[Any], name: str):
        return [(name, sig.answer) for sig in signatures]


class HistoryAdapter(BaseSignatureAdapter):
    """Replays ``AgentMessage.responses`` through ``dspy.History``."""

    def msg2sig(self, message: AgentMessage) -> List[ToonHistorySig]:
        responses = message.responses
        history = [
            {"question": responses[i][1], "answer": responses[i + 1][1]}
            for i in range(0, len(responses) - 1, 2)
        ]
        return [
            ToonHistorySig(
                history=dspy.History(messages=history),
                context=(message.context or {}).get("data", ""),
                question=message.query,
                answer="",
            )
        ]

    def sig2msg(self, signatures: List[ToonHistorySig], name: str):
        return [(name, sig.answer) for sig in signatures]


class PrefillTableAdapter(BaseSignatureAdapter):
    """Static (prefill) binding path: the table never passes through context."""

    def msg2sig(self, message: AgentMessage) -> List[TableSig]:
        return [
            TableSig(
                table=self._prefill_dict.get("table", ""),
                question=message.query,
                answer="",
            )
        ]

    def sig2msg(self, signatures: List[TableSig], name: str):
        return [(name, sig.answer) for sig in signatures]


class _MockRetrieve(dspy.Retrieve):
    """dspy.Retrieve over fixed passages; no embedding model, no network."""

    def __init__(self, passages: List[str], **kwargs):
        super().__init__(**kwargs)
        self._passages = passages

    def forward(self, query: str, k: int | None = None, **kwargs):
        return self._passages


class _MockEmbeddings(dspy.Embeddings):
    """dspy.Embeddings whose corpus lookup is answered from a fixed passage list."""

    def __init__(self, passages: List[str], **kwargs):
        def mock_embedder(texts):
            return [[0.0] * 8 for _ in texts]

        super().__init__(corpus=passages, embedder=mock_embedder, **kwargs)
        self._passages = passages

    def forward(self, query: str, **kwargs):
        return type("MockResult", (), {"passages": self._passages})()


def make_chain(
    signature: type[dspy.Signature] = ToonSig,
    adapter: BaseSignatureAdapter | None = None,
    reply: str = "It is a table.",
) -> ChatCausalMultiTurnsChain:
    """Build a chain wired to a recording LM, the way tests/test_chain.py does."""
    if adapter is None:
        adapter = HistoryAdapter() if signature is ToonHistorySig else ContextAdapter()
    chain = ChatCausalMultiTurnsChain(
        signature=signature,
        predictor=dspy.Predict(signature),
        adapter=adapter,
    )
    chain.with_lm(MockLM(reply=reply))
    return chain


def toon_dataframe(records: List[Dict[str, Any]], data_key: str = "context.data"):
    """A DataFrameRetriever whose payload is TOON-encoded, via the public factory."""
    return DataFrameRetriever.from_pandas(
        pd.DataFrame(records), data_key=data_key, prettier="toon"
    )


def messages_of(chain: ChatCausalMultiTurnsChain, index: int = -1) -> List[dict]:
    return chain._lm.captured[index]["messages"]


def user_content(chain: ChatCausalMultiTurnsChain, index: int = -1) -> str:
    users = [m for m in messages_of(chain, index) if m["role"] == "user"]
    assert users, "dspy must send at least one user message"
    content = users[-1]["content"]
    assert isinstance(content, str), "text-only signatures send a plain string"
    return content


def context_segment(chain: ChatCausalMultiTurnsChain, field: str = "context") -> str:
    """The exact bytes dspy placed in the named field of the last user message."""
    content = user_content(chain)
    marker = f"[[ ## {field} ## ]]\n"
    assert marker in content, f"user message lost the {field} marker: {content!r}"
    return content.split(marker, 1)[1].split("\n\n[[ ## ", 1)[0]


def occurrences(chain: ChatCausalMultiTurnsChain, needle: str) -> int:
    """Times the payload appears across every message handed to the LM."""
    return sum(str(m["content"]).count(needle) for m in messages_of(chain))


def resolved_aap_core_version() -> tuple:
    """Installed aap_core as a comparable tuple, tolerant of suffixes."""
    raw = version("aap_core")
    numbers: List[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple(numbers)


class TestResolvedCoreSupportsToon:
    """The upgrade claim: the aap_core this repo resolves actually enables TOON."""

    def test_toon_format_distribution_is_importable(self):
        """toon_format is importable and round-trips, so the core branch can run."""
        assert (
            toon_format.encode([{"name": "Ann", "age": 30}])
            == "[1]{name,age}:\n  Ann,30"
        )
        assert toon_format.decode("a: 1") == {"a": 1}
        assert toon_format.encode({"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}) == (
            "a: 1\nb[3]: 1,2,3\nc:\n  x: y"
        )

    def test_aap_core_version_meets_toon_requirement(self):
        """Resolved aap_core is >= 0.3.1, the release that enabled toon_format."""
        try:
            resolved = resolved_aap_core_version()
        except PackageNotFoundError as exc:  # pragma: no cover - env misconfig
            pytest.fail(f"aap_core is not installed in this environment: {exc}")
        assert resolved >= TOON_MIN_AAP_CORE, (
            f"aap_core {resolved} resolved from the environment does not enable "
            f"toon_format; aap_core>={'.'.join(map(str, TOON_MIN_AAP_CORE))} is "
            "required. Bump the dependency floor in pyproject.toml (it currently "
            "declares aap_core>=0.1.0) and re-run uv sync."
        )

    def test_declared_dependency_floor_enables_toon(self):
        """The manifest must promise a core that can actually emit TOON."""
        manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(
            r"aap_core\s*>=\s*([0-9]+(?:\.[0-9]+)*)", manifest.read_text()
        )
        assert match, f"no aap_core requirement found in {manifest}"
        declared = tuple(int(p) for p in match.group(1).split("."))
        assert declared >= TOON_MIN_AAP_CORE, (
            f"pyproject.toml requires aap_core>={'.'.join(map(str, declared))}, "
            f"which predates TOON support in {TOON_MIN_AAP_CORE}"
        )

    def test_from_pandas_routes_toon_to_toon_format(self):
        """prettier="toon" encodes records losslessly instead of falling to tabulate."""
        retriever = toon_dataframe(TABLE)
        message = retriever.retrieve(AgentMessage(query="q"))

        payload = message.context["data"]
        assert isinstance(payload, str), "the TOON branch must store a string"
        assert payload == PAYLOAD
        assert toon_format.decode(payload) == TABLE

    def test_from_dict_routes_toon_to_toon_format(self):
        """prettier="toon" on a dict encodes it losslessly."""
        data = {"warehouse": "north", "capacity": 128}
        retriever = DataFrameRetriever.from_dict(data, prettier="toon")
        message = retriever.retrieve(AgentMessage(query="q"))

        assert message.context["data"] == "warehouse: north\ncapacity: 128"
        assert toon_format.decode(message.context["data"]) == data

    def test_toon_keyword_is_case_sensitive_and_degrades_silently(self):
        """Only the exact lowercase "toon" routes to toon_format.

        A misspelled prettier is passed to tabulate as a tablefmt, which does not
        reject unknown formats. This is the trap this canary guards: a typo yields
        a plausible-looking string that is NOT TOON, with no exception raised.
        """
        typo = DataFrameRetriever.from_pandas(pd.DataFrame(TABLE), prettier="TOON")
        payload = typo.retrieve(AgentMessage(query="q")).context["data"]

        assert "{" not in payload, (
            "prettier='TOON' unexpectedly produced TOON output; the routing "
            "condition in aap_core changed and this canary needs revisiting"
        )
        assert payload != PAYLOAD
        with pytest.raises(toon_format.ToonDecodeError):
            toon_format.decode(payload)


class TestToonPayloadIntoDspyPrompt:
    """TOON context must reach the DSPy LM call verbatim."""

    def test_dataframe_toon_payload_reaches_the_lm_verbatim(self):
        chain = make_chain()
        message = toon_dataframe(TABLE).retrieve(AgentMessage(query="total qty?"))

        chain(message)

        assert context_segment(chain) == PAYLOAD, "TOON bytes must not change"
        assert user_content(chain).startswith(
            f"{CONTEXT_MARKER}{PAYLOAD}{QUESTION_MARKER}total qty?"
        )

    def test_toon_tabular_header_braces_survive_signature_interpolation(self):
        """TOON's "{cols}" header must not be consumed as a format placeholder."""
        assert "{" in PAYLOAD, (
            "TOON no longer emits brace-bearing headers; this regression canary "
            "no longer exercises the risk it was written for"
        )
        chain = make_chain()

        chain(AgentMessage(query="q", context={"data": PAYLOAD}))

        assert context_segment(chain) == PAYLOAD
        assert occurrences(chain, PAYLOAD) == 1, "payload must appear exactly once"

    def test_system_message_carries_placeholders_not_the_payload(self):
        """dspy keeps the field structure in the system message as literal braces.

        The payload is a value, never a template, so it must not be interpolated
        into the system message - and the braces there are dspy's own.
        """
        chain = make_chain()

        chain(AgentMessage(query="q", context={"data": PAYLOAD}))

        system = messages_of(chain)[0]["content"]
        assert "{context}" in system, "dspy's field-structure template changed shape"
        assert PAYLOAD not in system

    def test_str_input_field_keeps_toon_a_string(self):
        """A TOON string bound to a str-annotated InputField stays a str."""
        signature = ToonSig(context=PAYLOAD, question="q", answer="")

        assert isinstance(signature.context, str)
        assert signature.context == PAYLOAD
        assert signature.model_dump(exclude_none=True)["context"] == PAYLOAD

    def test_generate_response_round_trip_preserves_the_payload(self):
        """The model_dump -> Signature reconstruction inside _generate_response is lossless."""
        chain = make_chain()
        conversation = chain._prepare_conversation(
            AgentMessage(query="q", context={"data": PAYLOAD})
        )

        conversation, _, _, _ = chain._generate_response(conversation)

        assert conversation[-1].context == PAYLOAD, (
            "_generate_response re-quoted or truncated the TOON table"
        )

    def test_prefilled_toon_table_reaches_the_lm_verbatim(self):
        """The static (prefill) binding path is as faithful as the context path."""
        chain = make_chain(
            signature=TableSig,
            adapter=PrefillTableAdapter.with_prefill({"table": PAYLOAD}),
        )

        chain(AgentMessage(query="total qty?"))

        assert context_segment(chain, field="table") == PAYLOAD

    def test_from_dict_toon_payload_reaches_the_lm_verbatim(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
        payload = toon_format.encode(data)
        chain = make_chain()

        chain(
            DataFrameRetriever.from_dict(data, prettier="toon").retrieve(
                AgentMessage(query="inventory")
            )
        )

        assert context_segment(chain) == payload
        assert toon_format.decode(context_segment(chain)) == data

    @pytest.mark.parametrize(
        "records",
        [
            [{"sku": "A-1", "qty": 2}],
            [
                {"sku": "A-1", "qty": 2},
                {"sku": "B-2", "qty": 7},
                {"sku": "C-3", "qty": 0},
            ],
            [{"name": "café", "note": "你好"}],
            [{"id": 1, "meta": {"x": 1}}, {"id": 2, "meta": {"x": 2}}],
            [{"a": 1, "b": "line1\nline2"}],
        ],
        ids=["single-row", "multi-row", "unicode", "nested-column", "multiline-value"],
    )
    def test_various_tables_are_delivered_verbatim(self, records):
        payload = toon_format.encode(records)
        chain = make_chain()

        chain(AgentMessage(query="q", context={"data": payload}))

        assert context_segment(chain) == payload
        assert toon_format.decode(context_segment(chain)) == records

    def test_empty_dataframe_encodes_to_empty_table_token(self):
        """Edge case: zero rows must not blow up the prompt path."""
        retriever = DataFrameRetriever.from_pandas(
            pd.DataFrame(TABLE).iloc[0:0], prettier="toon"
        )
        message = retriever.retrieve(AgentMessage(query="q"))
        chain = make_chain()

        chain(message)

        assert message.context["data"] == "[]"
        assert context_segment(chain) == "[]"
        assert toon_format.decode("[]") == []

    def test_nan_becomes_null_and_stays_prompt_safe(self):
        """Known lossiness: NaN is encoded as TOON null, decoded as None."""
        records = [{"a": 1.0, "b": float("nan")}]
        payload = (
            toon_dataframe(records).retrieve(AgentMessage(query="q")).context["data"]
        )
        chain = make_chain()

        chain(AgentMessage(query="q", context={"data": payload}))

        assert payload == "[1]{a,b}:\n  1,null"
        assert toon_format.decode(context_segment(chain)) == [{"a": 1, "b": None}]


class TestRetrieverAdapterWithToon:
    """The DSPy-side retriever must not mangle TOON payloads."""

    def test_single_passage_is_stored_as_a_string(self):
        """Hit count must not change the stored type: one hit is a str."""
        message = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD])).retrieve(
            AgentMessage(query="q")
        )

        assert isinstance(message.context["data"], str), (
            "RetrieverAdapter regressed to a type that depends on hit count"
        )
        assert message.context["data"] == PAYLOAD

    def test_multiple_passages_are_newline_joined(self):
        """Two hits land as one string separated by a newline, not a space."""
        message = RetrieverAdapter(
            retriever=_MockRetrieve([PAYLOAD, "spare part X"])
        ).retrieve(AgentMessage(query="q"))

        assert isinstance(message.context["data"], str)
        assert message.context["data"] == f"{PAYLOAD}\nspare part X"

    def test_zero_passages_is_stored_as_an_empty_string(self):
        message = RetrieverAdapter(retriever=_MockRetrieve([])).retrieve(
            AgentMessage(query="q")
        )

        assert message.context["data"] == ""

    def test_toon_payload_merges_into_existing_context(self):
        adapter = RetrieverAdapter(retriever=_MockRetrieve(["first", "second"]))
        message = AgentMessage(query="q", context={"other_key": "existing_value"})

        result = adapter.retrieve(message)

        assert result.context["other_key"] == "existing_value"
        assert result.context["data"] == "first\nsecond"

    def test_custom_data_key_holds_the_toon_payload(self):
        adapter = RetrieverAdapter(
            retriever=_MockRetrieve(["passage one", "passage two"]),
            data_key="context.stock",
        )

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context == {"stock": "passage one\npassage two"}

    def test_embeddings_source_reads_passages_for_a_toon_payload(self):
        """The adapter takes .passages off an Embeddings result; the bytes survive."""
        adapter = RetrieverAdapter(retriever=_MockEmbeddings([PAYLOAD, "tail row"]))

        result = adapter.retrieve(AgentMessage(query="q"))

        assert result.context["data"] == f"{PAYLOAD}\ntail row"

    def test_data_key_without_context_prefix_is_rejected(self):
        with pytest.raises(ValueError, match="data_key must start with 'context.'"):
            RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD]), data_key="toon_data")

    def test_post_process_composes_a_toon_dataframe_retriever(self):
        """DSPy passages plus a core TOON table in one message context."""
        adapter = RetrieverAdapter(
            retriever=_MockRetrieve(["prose passage"]),
            post_process=toon_dataframe(
                [{"sku": "Z-9", "qty": 1}], data_key="context.table"
            ),
        )

        result = adapter(AgentMessage(query="q"))

        assert result.context["data"] == "prose passage"
        assert toon_format.decode(result.context["table"]) == [{"sku": "Z-9", "qty": 1}]

    def test_single_hit_binds_cleanly_to_a_str_input_field(self):
        """The documented adapter pattern must work for one hit, not only for two.

        A single passage used to be stored as a bare list, which a str-annotated
        dspy.InputField rejected outright. RetrieverAdapter now always stores a str,
        so the crash-on-one-hit is gone. Pinned positively so the regression cannot
        come back.
        """
        message = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD])).retrieve(
            AgentMessage(query="q")
        )
        chain = make_chain()

        chain(message)

        assert context_segment(chain) == PAYLOAD
        assert toon_format.decode(context_segment(chain)) == TABLE

    def test_single_hit_toon_table_is_decodable_after_template_interpolation(self):
        message = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD])).retrieve(
            AgentMessage(query="q")
        )

        rendered = "{query}\n{context_data}".format(**message.format_kwargs())

        assert toon_format.decode(rendered.split("\n", 1)[1]) == TABLE

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TOON is a single-root document format, so no separator makes "
            "table+prose decodable as one value. A newline join no longer glues the "
            "prose onto the last row ('B-2,7 spare part X'), but the decoder still "
            "rejects the trailing content: ToonDecodeError: line 4: Unexpected "
            "content after the root value. Fixing this needs a container format "
            "(one context key per passage), not a different join character."
        ),
    )
    def test_each_toon_passage_stays_a_separate_decodable_table(self):
        adapter = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD, "spare part X"]))
        chain = make_chain()

        chain(adapter.retrieve(AgentMessage(query="q")))

        assert toon_format.decode(context_segment(chain)) == TABLE

    def test_a_list_typed_input_field_now_rejects_the_payload(self):
        """Migration cost of the always-str contract, pinned explicitly.

        Signatures that worked around the old bare-list behaviour by typing the
        context field as list[str] - which dspy then blob-quoted into an
        undecodable '\u00ab\u00ab\u00ab ... \u00bb\u00bb\u00bb' block - now fail
        validation instead of silently corrupting the table. The correct field type
        is str. This test exists so that breakage is loud and documented rather
        than mysterious.
        """
        message = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD])).retrieve(
            AgentMessage(query="q")
        )
        chain = make_chain(
            signature=ToonListSig, adapter=ContextAdapter(field="list_context")
        )

        with pytest.raises(ValidationError, match="Input should be a valid list"):
            chain(message)


class TestToonEndToEnd:
    """Retriever -> chain -> AgentMessage, with the offline LM stub."""

    def test_retrieve_then_invoke_delivers_toon_to_the_model(self):
        records = [{"sku": "A-1", "qty": 2}, {"sku": "B-2", "qty": 7}]
        chain = make_chain(reply="9 units")

        result = chain(toon_dataframe(records)(AgentMessage(query="total qty?")))

        assert context_segment(chain) == PAYLOAD, "the model must see the raw table"
        assert occurrences(chain, PAYLOAD) == 1
        assert result.responses == [("chain", "9 units")]
        assert result.execution_result == "success"
        assert result.origin == "chain"
        assert result.context["data"] == PAYLOAD, "context must survive the chain"
        assert set(result.token_usage) == {"steps", "total"}
        assert len(result.token_usage["steps"]) == len(chain._lm.captured), (
            "one usage record per LM call; counts are zero because MockLM bypasses "
            "dspy's track_usage accounting, not because the step is missing"
        )
        assert result.token_usage["total"]["total_tokens"] == sum(
            step["total_tokens"] for step in result.token_usage["steps"]
        )

    def test_invoke_round_trips_toon_context_back_into_agent_message(self):
        chain = make_chain()

        message = chain.invoke(AgentMessage(query="q", context={"data": PAYLOAD}))

        assert toon_format.decode(message.context["data"]) == TABLE
        assert toon_format.decode(message.to_dict()["context_data"]) == TABLE
        assert toon_format.decode(message.format_kwargs()["context_data"]) == TABLE

    def test_toon_context_is_not_duplicated_through_replayed_history(self):
        """History replays as real dspy messages; the table belongs to this turn only."""
        chain = make_chain(signature=ToonHistorySig)
        chain.include_history = 2
        message = AgentMessage(
            query="again?",
            context={"data": PAYLOAD},
            responses=[("agent", "earlier question"), ("agent", "earlier answer")],
        )

        result = chain(message)

        roles = [m["role"] for m in messages_of(chain)]
        assert roles == ["system", "user", "assistant", "user"]
        assert occurrences(chain, PAYLOAD) == 1, (
            "the TOON table leaked into replayed history"
        )
        assert context_segment(chain) == PAYLOAD
        assert result.responses[-1] == ("chain", "It is a table.")

    def test_retriever_adapter_then_chain_delivers_the_joined_string_verbatim(self):
        """The chain layer is faithful even when the adapter layer already merged.

        Two hits arrive as one newline-joined string; the prompt reproduces exactly
        those bytes. Any remaining decode gap is the adapter's container model,
        upstream of dspy - which is what the strict xfail above pins down.
        """
        joined = f"{PAYLOAD}\nspare part X"
        chain = make_chain()
        message = RetrieverAdapter(retriever=_MockRetrieve([PAYLOAD, "spare part X"]))(
            AgentMessage(query="q")
        )

        result = chain(message)

        assert context_segment(chain) == joined
        assert occurrences(chain, PAYLOAD) == 1
        assert result.context["data"] == joined
        assert result.execution_result == "success"
