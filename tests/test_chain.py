"""Offline structural tests for aap_dspy.chain image, audio and video input support.

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
    Video,
    media_ref_to_audio,
    media_ref_to_image,
    media_ref_to_video,
    model_info_to_capabilities,
    rehydrate_audio_field,
    rehydrate_image_field,
    rehydrate_video_field,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# RIFF container with the WAVE form at offset 8: what aap_core's sniffer
# requires to identify a local file as audio/wav.
WAV_MAGIC = b"RIFF\x00\x00\x00\x00WAVE"
# ISO-BMFF file-size(4) + 'ftyp' + major_brand 'isom': what aap_core's sniffer
# requires to identify a local file as video/mp4.
MP4_MAGIC = b"\x00\x00\x00\x18ftyp" + b"isom" + b"\x00" * 8


class VisionSig(dspy.Signature):
    """Answer the question about the image."""

    image: Optional[dspy.Image] = dspy.InputField(default=None)
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class AudioSig(dspy.Signature):
    """Answer the question about the audio."""

    audio: Optional[dspy.Audio] = dspy.InputField(default=None)
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class VideoSig(dspy.Signature):
    """Answer the question about the video."""

    video: Optional[Video] = dspy.InputField(default=None)
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


class AudioAdapter(BaseSignatureAdapter):
    """Reference implementation of the media adapter pattern for audio."""

    def __init__(self):
        self.chain = None

    def msg2sig(self, message: AgentMessage) -> List[AudioSig]:
        audios = self.chain._audio_parts(message)
        return [
            AudioSig(
                question=message.query,
                audio=(media_ref_to_audio(audios[0]) if audios else None),
                answer="",
            )
        ]

    def sig2msg(self, signatures: List[AudioSig], name: str):
        return [(name, sig.answer) for sig in signatures]


class VideoAdapter(BaseSignatureAdapter):
    """Reference implementation of the media adapter pattern for video."""

    def __init__(self):
        self.chain = None

    def msg2sig(self, message: AgentMessage) -> List[VideoSig]:
        videos = self.chain._video_parts(message)
        return [
            VideoSig(
                question=message.query,
                video=(media_ref_to_video(videos[0]) if videos else None),
                answer="",
            )
        ]

    def sig2msg(self, signatures: List[VideoSig], name: str):
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


@pytest.fixture
def stub_audio_fetch(monkeypatch):
    """Serve dspy.Audio.from_url without network: URL refs are downloaded by
    design (Audio carries no URL field), so the fetch itself is stubbed."""
    calls: List[str] = []

    def fake_from_url(cls, url, **kwargs):
        calls.append(url)
        return cls(data="AAAA", audio_format="wav")

    monkeypatch.setattr(dspy.Audio, "from_url", classmethod(fake_from_url))
    return calls


def make_chain(signature=VisionSig, adapter=None, lm_model=None, **kwargs):
    if adapter is None:
        if signature is AudioSig:
            adapter = AudioAdapter()
        elif signature is VideoSig:
            adapter = VideoAdapter()
        else:
            adapter = VisionAdapter()
    chain = ChatCausalMultiTurnsChain(
        signature=signature,
        predictor=dspy.Predict(signature),
        adapter=adapter,
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

    def test_no_audio_field_populated_without_media(self):
        chain = make_chain(signature=AudioSig)
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0].audio is None

    def test_no_video_field_populated_without_media(self):
        chain = make_chain(signature=VideoSig)
        conversation = chain._prepare_conversation(AgentMessage(query="hello"))
        assert conversation[0].video is None


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


class TestMediaRefToAudio:
    """Direct tests of the MediaRef -> dspy.Audio mapping."""

    def test_base64_ref_becomes_audio(self):
        audio = media_ref_to_audio(
            {
                "content_type": "audio",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "audio/wav",
            }
        )
        assert isinstance(audio, dspy.Audio)
        assert audio.data == "AAAA"
        assert audio.audio_format == "wav"

    def test_x_prefixed_subtype_is_normalized(self):
        """Format token follows dspy's own convention (Audio.from_file strips
        the x- prefix), so a ref maps to the token dspy derives from the
        identical file."""
        audio = media_ref_to_audio(
            {
                "content_type": "audio",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "audio/x-ms-wma",
            }
        )
        assert audio.audio_format == "ms-wma"

    def test_url_ref_downloads_via_from_url(self, stub_audio_fetch):
        audio = media_ref_to_audio(
            {
                "content_type": "audio",
                "kind": "url",
                "value": "https://example.com/a.wav",
                "mime_type": "audio/wav",
            }
        )
        # Audio has no URL field: the bytes must be inline, hence the fetch.
        assert stub_audio_fetch == ["https://example.com/a.wav"]
        assert audio.data == "AAAA"


class TestMediaRefToVideo:
    """Direct tests of the MediaRef -> Video mapping."""

    def test_base64_ref_becomes_data_uri(self):
        vid = media_ref_to_video(
            {
                "content_type": "video",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "video/mp4",
            }
        )
        assert isinstance(vid, Video)
        assert vid.url == "data:video/mp4;base64,AAAA"

    def test_url_ref_passes_url_through(self):
        vid = media_ref_to_video(
            {
                "content_type": "video",
                "kind": "url",
                "value": "https://example.com/a.mp4",
                "mime_type": "video/mp4",
            }
        )
        # Unlike Audio, Video carries a URL field: no download at mapping time.
        assert vid.url == "https://example.com/a.mp4"

    def test_format_emits_file_content_part(self):
        vid = media_ref_to_video(
            {
                "content_type": "video",
                "kind": "base64",
                "value": "AAAA",
                "mime_type": "video/quicktime",
            }
        )
        parts = vid.format()
        assert parts == [
            {"type": "file", "file": {"file_data": "data:video/quicktime;base64,AAAA"}}
        ]


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


class TestRehydrateAudioField:
    """The model_dump round-trip fix for Audio fields."""

    def test_marker_string_rehydrates_to_audio(self):
        original = dspy.Audio(data="AAAA", audio_format="wav")
        dumped = original.serialize_model()
        assert isinstance(dumped, str) and dumped.startswith("<<CUSTOM-TYPE")
        restored = rehydrate_audio_field(dumped)
        assert isinstance(restored, dspy.Audio)
        assert restored.data == "AAAA"
        assert restored.audio_format == "wav"

    def test_audio_object_passes_through(self):
        audio = dspy.Audio(data="AAAA", audio_format="flac")
        assert rehydrate_audio_field(audio) is audio

    @pytest.mark.parametrize(
        "bad",
        [
            "plain text",
            "<<CUSTOM-TYPE-START-IDENTIFIER>>not json<<CUSTOM-TYPE-END-IDENTIFIER>>",
            # An image marker is not audio: silently rebuilding it as Audio
            # would send image bytes on the input_audio channel.
            dspy.Image(url="https://example.com/a.png").serialize_model(),
        ],
    )
    def test_bad_input_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            rehydrate_audio_field(bad)


class TestRehydrateVideoField:
    """The model_dump round-trip fix for Video fields."""

    def test_marker_string_rehydrates_to_video(self):
        original = Video(url="data:video/mp4;base64,AAAA")
        dumped = original.serialize_model()
        assert isinstance(dumped, str) and dumped.startswith("<<CUSTOM-TYPE")
        restored = rehydrate_video_field(dumped)
        assert isinstance(restored, Video)
        assert restored.url == "data:video/mp4;base64,AAAA"

    def test_url_marker_rehydrates(self):
        original = Video(url="https://example.com/a.mp4")
        restored = rehydrate_video_field(original.serialize_model())
        assert restored.url == "https://example.com/a.mp4"

    def test_video_object_passes_through(self):
        vid = Video(url="data:video/webm;base64,BBBB")
        assert rehydrate_video_field(vid) is vid

    @pytest.mark.parametrize(
        "bad",
        [
            "plain text",
            "<<CUSTOM-TYPE-START-IDENTIFIER>>not json<<CUSTOM-TYPE-END-IDENTIFIER>>",
            # An image marker is not video: silently rebuilding it as Video
            # would send image bytes on the file channel.
            dspy.Image(url="https://example.com/a.png").serialize_model(),
            # An audio marker is not video either.
            dspy.Audio(data="AAAA", audio_format="wav").serialize_model(),
        ],
    )
    def test_bad_input_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            rehydrate_video_field(bad)


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

    def test_audio_survives_dump_into_predictor(self):
        chain = make_chain(signature=AudioSig)
        msg = AgentMessage(
            query="what?",
            query_media=[("audio", "data:audio/wav;base64,AAAA")],
        )
        conversation = chain._prepare_conversation(msg)
        conversation, response, has_tool, usage = chain._generate_response(conversation)
        assert response.answer == "It is a cat."
        assert not has_tool
        parts = user_content_parts(chain._lm)
        assert any(p["type"] == "input_audio" for p in parts)
        # the appended conversation entry keeps a real Audio, not a marker string
        assert isinstance(conversation[-1].audio, dspy.Audio)
        assert conversation[-1].audio.data == "AAAA"

    def test_video_survives_dump_into_predictor(self):
        chain = make_chain(signature=VideoSig)
        msg = AgentMessage(
            query="what?",
            query_media=[("video", "data:video/mp4;base64,AAAA")],
        )
        conversation = chain._prepare_conversation(msg)
        conversation, response, has_tool, usage = chain._generate_response(conversation)
        assert response.answer == "It is a cat."
        assert not has_tool
        parts = user_content_parts(chain._lm)
        assert any(p["type"] == "file" for p in parts)
        # the appended conversation entry keeps a real Video, not a marker string
        assert isinstance(conversation[-1].video, Video)
        assert conversation[-1].video.url == "data:video/mp4;base64,AAAA"


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

    def test_audio_reaches_lm(self):
        chain = make_chain(signature=AudioSig)
        msg = AgentMessage(
            query="transcribe", query_media=[("audio", "data:audio/wav;base64,AAAA")]
        )
        out = chain(msg)
        assert out.responses[-1][1] == "It is a cat."
        parts = user_content_parts(chain._lm)
        audio_parts = [p for p in parts if p["type"] == "input_audio"]
        assert len(audio_parts) == 1
        assert audio_parts[0]["input_audio"]["data"] == "AAAA"
        assert audio_parts[0]["input_audio"]["format"] == "wav"

    def test_local_path_audio_reaches_lm(self, tmp_path):
        f = tmp_path / "speech.wav"
        f.write_bytes(WAV_MAGIC + b"\x00" * 16)
        chain = make_chain(signature=AudioSig)
        msg = AgentMessage(query="listen", query_media=[("audio", str(f))])
        chain(msg)
        parts = user_content_parts(chain._lm)
        audio_parts = [p for p in parts if p["type"] == "input_audio"]
        assert audio_parts[0]["input_audio"]["format"] == "wav"
        payload = audio_parts[0]["input_audio"]["data"]
        assert base64.b64decode(payload) == f.read_bytes()

    def test_url_audio_downloads_then_reaches_lm(self, stub_audio_fetch):
        chain = make_chain(signature=AudioSig)
        msg = AgentMessage(
            query="listen", query_media=[("audio", "https://example.com/a.wav")]
        )
        chain(msg)
        assert stub_audio_fetch == ["https://example.com/a.wav"]
        parts = user_content_parts(chain._lm)
        assert any(p["type"] == "input_audio" for p in parts)

    def test_text_only_message_sends_no_audio_parts(self):
        chain = make_chain(signature=AudioSig)
        chain(AgentMessage(query="hello"))
        parts = user_content_parts(chain._lm)
        assert not any(p["type"] == "input_audio" for p in parts)

    def test_video_reaches_lm_as_file_part(self):
        chain = make_chain(signature=VideoSig)
        msg = AgentMessage(
            query="describe", query_media=[("video", "data:video/mp4;base64,AAAA")]
        )
        out = chain(msg)
        assert out.responses[-1][1] == "It is a cat."
        parts = user_content_parts(chain._lm)
        file_parts = [p for p in parts if p["type"] == "file"]
        assert len(file_parts) == 1
        assert file_parts[0]["file"]["file_data"] == "data:video/mp4;base64,AAAA"

    def test_local_path_video_reaches_lm_as_data_uri(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(MP4_MAGIC + b"\x00" * 16)
        chain = make_chain(signature=VideoSig)
        msg = AgentMessage(query="describe", query_media=[("video", str(f))])
        chain(msg)
        parts = user_content_parts(chain._lm)
        file_parts = [p for p in parts if p["type"] == "file"]
        assert file_parts[0]["file"]["file_data"].startswith("data:video/mp4;base64,")
        payload = file_parts[0]["file"]["file_data"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == f.read_bytes()

    def test_url_video_passes_url_through(self):
        """Unlike audio, video does not download at mapping time."""
        chain = make_chain(signature=VideoSig)
        msg = AgentMessage(
            query="describe", query_media=[("video", "https://example.com/a.mp4")]
        )
        chain(msg)
        parts = user_content_parts(chain._lm)
        file_parts = [p for p in parts if p["type"] == "file"]
        assert file_parts[0]["file"]["file_data"] == "https://example.com/a.mp4"

    def test_text_only_message_sends_no_video_parts(self):
        chain = make_chain(signature=VideoSig)
        chain(AgentMessage(query="hello"))
        parts = user_content_parts(chain._lm)
        assert not any(p["type"] == "file" for p in parts)


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
            # Unmapped revisions stay undecidable. The name must match no
            # litellm pattern, present or future: real dated revisions rot
            # (litellm added claude-sonnet-4-20250514, then a claude-sonnet-4-*
            # wildcard, to its table), so the token is deliberately fake while
            # keeping the provider/model shape the code parses.
            ("anthropic/this-model-will-never-exist", {}),
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

    def test_video_is_never_reported_from_litellm(self):
        """litellm has no supports_video_input flag; video must stay absent
        from the capabilities dict so auto mode fails loud rather than
        silently denying."""
        info = {"supports_vision": True, "supports_audio_input": True}
        caps = model_info_to_capabilities(info)
        assert "video" not in caps


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

    def test_non_audio_model_drops_audio_with_warning(self, caplog, hf):
        hf.results["meta-llama/Llama-3.2-3B-Instruct"] = {"audio": False}
        chain = make_chain(
            signature=AudioSig,
            lm_model="hosted_vllm/meta-llama/Llama-3.2-3B-Instruct",
        )
        msg = AgentMessage(
            query="listen", query_media=[("audio", "data:audio/wav;base64,AAAA")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            conversation = chain._prepare_conversation(msg)
        assert conversation[0].audio is None  # degraded to text-only, no exception
        assert "does not support audio input" in caplog.text

    def test_disabled_policy_overrides_audio_detection(self):
        chain = make_chain(
            signature=AudioSig,
            lm_model="openai/gpt-4o-mini",
            media_support={"audio": "disabled"},
        )
        msg = AgentMessage(
            query="listen", query_media=[("audio", "data:audio/wav;base64,AAAA")]
        )
        assert chain._prepare_conversation(msg)[0].audio is None

    def test_enabled_policy_forces_audio_on_unknown_model(self):
        chain = make_chain(
            signature=AudioSig,
            lm_model="provider/some-brand-new-model",
            media_support={"audio": "enabled"},
        )
        msg = AgentMessage(
            query="listen", query_media=[("audio", "data:audio/wav;base64,AAAA")]
        )
        audio = chain._prepare_conversation(msg)[0].audio
        assert isinstance(audio, dspy.Audio)
        assert audio.audio_format == "wav"

    def test_non_video_model_drops_video_with_warning(self, caplog, hf):
        hf.results["meta-llama/Llama-3.2-3B-Instruct"] = {"video": False}
        chain = make_chain(
            signature=VideoSig,
            lm_model="hosted_vllm/meta-llama/Llama-3.2-3B-Instruct",
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "data:video/mp4;base64,AAAA")]
        )
        with caplog.at_level(logging.WARNING, logger="aap_core.types"):
            conversation = chain._prepare_conversation(msg)
        assert conversation[0].video is None
        assert "does not support video input" in caplog.text

    def test_disabled_policy_overrides_video_detection(self):
        chain = make_chain(
            signature=VideoSig,
            lm_model="openai/gpt-4o-mini",
            media_support={"video": "disabled"},
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "data:video/mp4;base64,AAAA")]
        )
        assert chain._prepare_conversation(msg)[0].video is None

    def test_enabled_policy_forces_video_on_unknown_model(self):
        chain = make_chain(
            signature=VideoSig,
            lm_model="provider/some-brand-new-model",
            media_support={"video": "enabled"},
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "data:video/mp4;base64,AAAA")]
        )
        vid = chain._prepare_conversation(msg)[0].video
        assert isinstance(vid, Video)
        assert vid.url == "data:video/mp4;base64,AAAA"

    def test_undecidable_model_attaches_video_fail_loud(self):
        chain = make_chain(
            signature=VideoSig,
            lm_model="provider/some-brand-new-model",
        )
        msg = AgentMessage(
            query="watch", query_media=[("video", "data:video/mp4;base64,AAAA")]
        )
        vid = chain._prepare_conversation(msg)[0].video
        assert isinstance(vid, Video)
