"""Offline structural tests for aap_dspy.chain image input support.

No network or real model calls: tests exercise the adapter funnel, the
model_dump round-trip rehydration, capability detection and the media gate.
The LM is a dspy.LM subclass whose __call__ only records the messages the
adapter produced, so assertions are on the native content parts dspy would
send to a provider. Capability detection reads litellm's static model-info
map (offline) and the Hugging Face model-card lookup, which every test stubs
out via the no_network/hf fixtures.
"""

import base64
import logging
from typing import Dict, List, Optional

import dspy
import pytest
from aap_core.types import AgentMessage, ContentType

import aap_dspy.chain as chain_module
from aap_dspy.chain import BaseSignatureAdapter, ChatCausalMultiTurnsChain
from aap_dspy.utils import (
    media_ref_to_image,
    model_info_to_capabilities,
    rehydrate_image_field,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class VisionSig(dspy.Signature):
    """Answer the question about the image."""

    image: Optional[dspy.Image] = dspy.InputField(default=None)
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class TextSig(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MockLM(dspy.LM):
    """LM stub that records adapter output instead of calling a provider."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        super().__init__(model, cache=False, max_tokens=16)
        self.captured: List[dict] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.captured.append({"messages": messages, "kwargs": kwargs})
        return ["[[ ## answer ## ]]\nIt is a cat.\n[[ ## completed ## ]]"]


class VisionAdapter(BaseSignatureAdapter):
    """Reference implementation of the documented media adapter pattern."""

    def __init__(self):
        self.chain = None

    def msg2sig(self, message: AgentMessage) -> List[VisionSig]:
        images = self.chain._image_parts(message)
        return [
            VisionSig(
                question=message.query,
                image=(media_ref_to_image(images[0]) if images else None),
                answer="",
            )
        ]

    def sig2msg(self, signatures: List[VisionSig], name: str):
        return [(name, sig.answer) for sig in signatures]


class TextAdapter(BaseSignatureAdapter):
    """Adapter for a text-only signature; must be unaffected by media support."""

    def msg2sig(self, message: AgentMessage) -> List[TextSig]:
        return [TextSig(question=message.query, answer="")]

    def sig2msg(self, signatures: List[TextSig], name: str):
        return [(name, sig.answer) for sig in signatures]


class HuggingFaceStub:
    """Recording stand-in for the model-card lookup; no test reaches the Hub."""

    def __init__(self):
        self.results: Dict[str, Dict[ContentType, bool]] = {}
        self.calls: List[str] = []

    def __call__(self, repo_id: str) -> Dict[ContentType, bool]:
        self.calls.append(repo_id)
        return self.results.get(repo_id, {})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Keep the default lookup inert so no test can reach the Hub."""
    monkeypatch.setattr(chain_module, "huggingface_capabilities", lambda repo_id: {})


@pytest.fixture
def hf(monkeypatch):
    """Install a recording Hugging Face stub over no_network's default."""
    stub = HuggingFaceStub()
    monkeypatch.setattr(chain_module, "huggingface_capabilities", stub)
    return stub


def make_chain(signature=VisionSig, adapter=None, lm_model=None, **kwargs):
    chain = ChatCausalMultiTurnsChain(
        signature=signature,
        predictor=dspy.Predict(signature),
        adapter=adapter if adapter is not None else VisionAdapter(),
        **kwargs,
    )
    chain.adapter.chain = chain
    lm = MockLM(lm_model) if lm_model else MockLM()
    chain.with_lm(lm)
    return chain


def user_content_parts(lm: MockLM, index: int = -1) -> list:
    messages = lm.captured[index]["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    content = user_msgs[-1]["content"]
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


class TestPrepareConversationTextOnly:
    """Regression: text-only behavior must be identical to before media support."""

    def test_adapter_receives_message(self):
        chain = make_chain(adapter=TextAdapter(), lm_model="ollama_chat/llama3.2")
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert len(conversation) == 1
        assert conversation[0].question == "hello"

    def test_no_image_field_populated_without_media(self):
        chain = make_chain()
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0].image is None


class TestMediaRefToImage:
    """Direct tests of the MediaRef -> dspy.Image mapping."""

    def test_url_ref_passes_url_through(self):
        img = media_ref_to_image(
            {
                "content_type": "image",
                "kind": "url",
                "value": "https://example.com/a.png",
                "mime_type": "image/png",
            }
        )
        assert img.url == "https://example.com/a.png"

    def test_base64_ref_becomes_data_uri(self):
        img = media_ref_to_image(
            {
                "content_type": "image",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "image/jpeg",
            }
        )
        assert img.url == "data:image/jpeg;base64,AAAA"


class TestRehydrateImageField:
    """The model_dump round-trip fix."""

    def test_marker_string_rehydrates_to_image(self):
        original = dspy.Image(url="https://example.com/a.png")
        dumped = original.serialize_model()
        assert isinstance(dumped, str) and dumped.startswith("<<CUSTOM-TYPE")
        restored = rehydrate_image_field(dumped)
        assert isinstance(restored, dspy.Image)
        assert restored.url == "https://example.com/a.png"

    def test_data_uri_marker_rehydrates(self):
        original = dspy.Image(url="data:image/png;base64,AAAA")
        restored = rehydrate_image_field(original.serialize_model())
        assert restored.url == "data:image/png;base64,AAAA"

    def test_image_object_passes_through(self):
        img = dspy.Image(url="https://example.com/a.png")
        assert rehydrate_image_field(img).url == img.url

    @pytest.mark.parametrize(
        "bad",
        [
            "plain text",
            "<<CUSTOM-TYPE-START-IDENTIFIER>>not json<<CUSTOM-TYPE-END-IDENTIFIER>>",
        ],
    )
    def test_bad_input_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            rehydrate_image_field(bad)


class TestGenerateResponseRoundTrip:
    """_generate_response must not degrade Image fields to marker strings."""

    def test_image_survives_dump_into_predictor(self):
        chain = make_chain()
        msg = AgentMessage(
            query="what?", query_media=[("image", "https://example.com/cat.png")]
        )
        conversation = chain._prepare_conversation(msg)
        conversation, response, has_tool, usage = chain._generate_response(conversation)
        assert response.answer == "It is a cat."
        assert not has_tool
        parts = user_content_parts(chain._lm)
        assert any(p["type"] == "image_url" for p in parts)
        # the appended conversation entry keeps a real Image, not a marker string
        assert isinstance(conversation[-1].image, dspy.Image)
        assert conversation[-1].image.url == "https://example.com/cat.png"

    def test_text_only_signature_unaffected(self):
        chain = make_chain(signature=TextSig, adapter=TextAdapter())
        msg = AgentMessage(query="what?")
        conversation = chain._prepare_conversation(msg)
        conversation, response, has_tool, _ = chain._generate_response(conversation)
        assert response.answer == "It is a cat."
        assert not any(p["type"] == "image_url" for p in user_content_parts(chain._lm))


class TestInvokeEndToEnd:
    """Full invoke loop: media must reach the LM as a native content part."""

    def test_image_reaches_lm(self):
        chain = make_chain()
        msg = AgentMessage(
            query="describe", query_media=[("image", "https://example.com/cat.png")]
        )
        out = chain(msg)
        assert out.responses[-1][1] == "It is a cat."
        parts = user_content_parts(chain._lm)
        image_parts = [p for p in parts if p["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"] == "https://example.com/cat.png"

    def test_local_path_image_reaches_lm_as_data_uri(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(PNG_MAGIC + b"\x00" * 16)
        chain = make_chain()
        msg = AgentMessage(query="describe", query_media=[("image", str(f))])
        chain(msg)
        parts = user_content_parts(chain._lm)
        image_parts = [p for p in parts if p["type"] == "image_url"]
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        payload = image_parts[0]["image_url"]["url"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == f.read_bytes()

    def test_text_only_message_sends_no_image_parts(self):
        chain = make_chain()
        chain(AgentMessage(query="hello"))
        parts = user_content_parts(chain._lm)
        assert not any(p["type"] == "image_url" for p in parts)


class TestCapabilityDetection:
    """Metadata-based detection: litellm model-info table, then HF model cards."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            # litellm's static table answers for mapped provider revisions.
            ("openai/gpt-4o-mini", {"image": True, "document": True}),
            ("azure/gpt-4o", {"image": True}),
            # Mapped but with no input-modality flags: undecidable, not denial.
            ("openai/gpt-3.5-turbo", {}),
            # Unmapped revisions stay undecidable.
            ("anthropic/claude-sonnet-4-20250514", {}),
            ("provider/some-brand-new-model-v9", {}),
        ],
    )
    def test_detect_capabilities(self, name, expected):
        chain = make_chain(lm_model=name)
        assert chain.detect_capabilities() == expected

    def test_family_name_alone_is_not_evidence(self, monkeypatch):
        """The lesson from the name heuristics: "gemma3" must not report True.

        get_model_info is pinned to raise so the case cannot depend on
        whether a local ollama server happens to answer for the name.
        """

        def unmapped(model=None, **kwargs):
            raise Exception("This model isn't mapped yet.")

        monkeypatch.setattr(chain_module.litellm, "get_model_info", unmapped)
        chain = make_chain(lm_model="ollama_chat/gemma3:12b")
        assert chain.detect_capabilities() == {}

    def test_repo_id_model_falls_back_to_hugging_face(self, hf):
        hf.results["Qwen/Qwen2-VL-2B-Instruct"] = {"image": True}
        chain = make_chain(lm_model="hosted_vllm/Qwen/Qwen2-VL-2B-Instruct")
        assert chain.detect_capabilities() == {"image": True}
        assert hf.calls == ["Qwen/Qwen2-VL-2B-Instruct"]

    def test_provider_names_are_not_looked_up_on_hugging_face(self, hf):
        chain = make_chain(lm_model="ollama_chat/gemma3:12b")
        chain.detect_capabilities()
        assert hf.calls == []

    def test_result_is_cached_per_lm(self, hf):
        hf.results["google/gemma-3-4b-it"] = {"image": True}
        chain = make_chain(lm_model="hosted_vllm/google/gemma-3-4b-it")
        chain.detect_capabilities()
        chain.detect_capabilities()
        assert hf.calls == ["google/gemma-3-4b-it"]

    def test_cache_invalidates_when_the_lm_is_replaced(self, hf):
        hf.results["google/gemma-3-4b-it"] = {"image": True}
        chain = make_chain(lm_model="hosted_vllm/google/gemma-3-4b-it")
        assert chain.detect_capabilities() == {"image": True}
        chain.with_lm(MockLM("hosted_vllm/openai/gpt-oss-120b"))
        chain.detect_capabilities()
        assert hf.calls == ["google/gemma-3-4b-it", "openai/gpt-oss-120b"]

    def test_no_lm_configured_is_undecidable(self):
        chain = ChatCausalMultiTurnsChain(
            signature=VisionSig,
            predictor=dspy.Predict(VisionSig),
            adapter=VisionAdapter(),
        )
        chain.adapter.chain = chain
        assert chain._active_lm() is None
        assert chain.detect_capabilities() == {}


class TestModelInfoToCapabilities:
    """Direct tests of the litellm info -> modality mapping."""

    def test_boolean_flags_are_mapped(self):
        info = {"supports_vision": True, "supports_audio_input": False}
        assert model_info_to_capabilities(info) == {"image": True, "audio": False}

    def test_missing_and_none_flags_stay_undecidable(self):
        assert model_info_to_capabilities({"supports_vision": None}) == {}

    def test_non_dict_info_is_undecidable(self):
        assert model_info_to_capabilities(None) == {}


class TestGateIntegration:
    """End-to-end of the core gate through the dspy adapter funnel."""

    def test_text_only_model_drops_images_with_warning(self, caplog, hf):
        hf.results["meta-llama/Llama-3.2-3B-Instruct"] = {"image": False}
        chain = make_chain(lm_model="hosted_vllm/meta-llama/Llama-3.2-3B-Instruct")
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            conversation = chain._prepare_conversation(msg)
        assert conversation[0].image is None  # degraded to text-only, no exception
        assert "does not support image input" in caplog.text

    def test_disabled_policy_overrides_vision_detection(self):
        chain = make_chain(
            lm_model="ollama_chat/gemma3", media_support={"image": "disabled"}
        )
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        assert chain._prepare_conversation(msg)[0].image is None

    def test_enabled_policy_forces_image_on_unknown_model(self):
        chain = make_chain(
            lm_model="provider/some-brand-new-model",
            media_support={"image": "enabled"},
        )
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        img = chain._prepare_conversation(msg)[0].image
        assert isinstance(img, dspy.Image)
        assert img.url == "https://example.com/a.png"

    def test_undecidable_model_attaches_image_fail_loud(self):
        chain = make_chain(lm_model="provider/some-brand-new-model")
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        assert isinstance(chain._prepare_conversation(msg)[0].image, dspy.Image)

    def test_non_vision_model_invoke_succeeds_text_only(self, hf):
        hf.results["meta-llama/Llama-3.2-3B-Instruct"] = {"image": False}
        chain = make_chain(lm_model="hosted_vllm/meta-llama/Llama-3.2-3B-Instruct")
        msg = AgentMessage(
            query="look", query_media=[("image", "https://example.com/a.png")]
        )
        out = chain(msg)
        assert out.responses[-1][1] == "It is a cat."
        parts = user_content_parts(chain._lm)
        assert not any(p["type"] == "image_url" for p in parts)

    def test_missing_local_image_raises(self, tmp_path):
        chain = make_chain(lm_model="ollama_chat/gemma3")
        msg = AgentMessage(
            query="look", query_media=[("image", str(tmp_path / "nope.png"))]
        )
        with pytest.raises(FileNotFoundError):
            chain._prepare_conversation(msg)
