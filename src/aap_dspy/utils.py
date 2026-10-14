import json
from typing import Any, Dict, List

from aap_core.types import ContentType, MediaReference, TokenUsage
from dspy import Audio, Image, Prediction, Type
from dspy.adapters.types.base_type import (
    CUSTOM_TYPE_END_IDENTIFIER,
    CUSTOM_TYPE_START_IDENTIFIER,
)

# litellm model-info keys, mapped onto the unified ContentType. The table is a
# dict of optional flags, so any key may be absent; absent keys stay
# undecidable rather than being reported as unsupported. litellm has no video
# input flag, so "video" is never reported from this source.
_LITELLM_INPUT_KEYS: Dict[ContentType, str] = {
    "image": "supports_vision",
    "audio": "supports_audio_input",
    "document": "supports_pdf_input",
    "video": "supports_video_input",
}


class Video(Type):
    """Video input type for DSPy signatures.

    dspy has no native Video type (none through 3.4.1, checked 2026-10; the
    upstream dspy.Video PRs #9369/#9442/#9443 were all closed unmerged). This
    subclass fills the gap by emitting a ``file`` content part, which is the
    wire format litellm's provider handlers already dispatch for non-image
    binary media: Gemini's
    ``_process_gemini_image`` converts it to ``inline_data`` (base64) or
    ``file_data`` (URL) with the correct MIME type, and providers without video
    input reject it at the API level (fail loud, same as audio on a
    non-audio model).

    The ``url`` field carries either a remote URL or a ``data:`` URI, mirroring
    ``dspy.Image.url``. Local file paths never reach here: aap_core.resolve_media
    already read and base64-encoded them.
    """

    url: str

    def format(self) -> List[Dict[str, Any]]:
        return [{"type": "file", "file": {"file_data": self.url}}]


def token_from_response(prediction: Prediction) -> TokenUsage:
    """
    Convert dspy's prediction object to a TokenUsage object.

    Args:
        predicted (Prediction): The prediction object to convert.

    Returns:
        TokenUsage: The converted object.
    """
    usage = prediction.get_lm_usage()
    if usage is not None and len(usage) > 0:
        usage = list(usage.values())[0]
        return TokenUsage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
    return TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)


def model_info_to_capabilities(info: Any) -> Dict[ContentType, bool]:
    """Map a litellm get_model_info result onto gated input modalities.

    Only boolean flags count: None or missing stays undecidable, so auto mode
    keeps its fail-loud behavior instead of denying media the table simply
    does not describe.

    Args:
        info: The dict returned by litellm.get_model_info; non-dicts yield an
            empty result.

    Returns:
        Dict[ContentType, bool]: One entry per modality the table reports on.
    """
    if not isinstance(info, dict):
        return {}
    capabilities: Dict[ContentType, bool] = {}
    for modality, key in _LITELLM_INPUT_KEYS.items():
        value = info.get(key)
        if isinstance(value, bool):
            capabilities[modality] = value
    return capabilities


def media_ref_to_image(ref: MediaReference) -> Image:
    """Convert a resolved MediaReference to a dspy.Image.

    dspy.Image normalizes its ``url`` field at construction time:
    - kind="url" refs pass the URL through to the LM as-is.
    - kind="base64" refs become a ``data:`` URI, which dspy keeps verbatim.
    Local file paths never reach here: aap_core.resolve_media already read and
    base64-encoded them.

    Args:
        ref (MediaReference): The resolved media reference from
            BaseLLMChain._media_parts. Only image refs are mapped today.

    Returns:
        Image: A dspy.Image ready to be assigned to an InputField of a signature.
    """
    if ref["kind"] == "url":
        return Image(url=ref["value"])
    return Image(url=f"data:{ref['mime_type']};base64,{ref['value']}")


def media_ref_to_audio(ref: MediaReference) -> Audio:
    """Convert a resolved MediaReference to a dspy.Audio.

    Unlike dspy.Image, dspy.Audio carries no URL: its wire shape is OpenAI
    chat's ``input_audio`` part, which needs inline base64 data plus a short
    format token. A URL ref is therefore downloaded here, via the explicit
    ``Audio.from_url`` factory - dspy 3.3 removed the implicit fetch-on-string
    construction, so relying on it would break on upgrade.

    The format token follows dspy's own convention (MIME subtype with any
    "x-" prefix stripped, as in Audio.from_file), so a ref maps to the same
    token dspy would derive from the identical file. The token is not
    rewritten beyond that: whether a provider accepts it (OpenAI chat takes
    only wav/mp3) is provider knowledge below this layer, and
    SUPPORTED_MIME_TYPES["audio"] in aap_core is the set refs arrive in.

    Args:
        ref (MediaReference): The resolved media reference from
            BaseLLMChain._audio_parts.

    Returns:
        Audio: A dspy.Audio ready to be assigned to an InputField of a signature.
    """
    if ref["kind"] == "url":
        return Audio.from_url(ref["value"])
    subtype = ref["mime_type"].partition("/")[2]
    return Audio(data=ref["value"], audio_format=subtype.removeprefix("x-"))


def media_ref_to_video(ref: MediaReference) -> Video:
    """Convert a resolved MediaReference to a Video.

    Unlike dspy.Audio (which carries no URL and must download at mapping time),
    Video mirrors dspy.Image: a URL ref passes through as-is and a base64 ref
    becomes a data URI. The provider decides whether to fetch server-side or
    receive inline bytes; this layer does not download.

    Args:
        ref (MediaReference): The resolved media reference from
            BaseLLMChain._video_parts.

    Returns:
        Video: A Video ready to be assigned to an InputField of a signature.
    """
    if ref["kind"] == "url":
        return Video(url=ref["value"])
    return Video(url=f"data:{ref['mime_type']};base64,{ref['value']}")


def rehydrate_image_field(serialized: Any) -> Image:
    """Rebuild a dspy.Image from the marker string produced by model_dump.

    A dspy.Image input field survives Signature.model_dump() only as a
    ``<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "image_url", "image_url":
    {"url": ...}}]<<CUSTOM-TYPE-END-IDENTIFIER>>`` string. Feeding that string
    back into a signature or predictor degrades the image to text garbage, so
    it must be converted to a real Image object before the predictor call.

    Args:
        serialized: The dumped value of an Image-typed field.

    Returns:
        Image: The rehydrated image object.

    Raises:
        ValueError: If the value is not a parseable dspy custom-type marker.
    """
    if isinstance(serialized, Image):
        return serialized
    if isinstance(serialized, str) and serialized.startswith(
        CUSTOM_TYPE_START_IDENTIFIER
    ):
        body = serialized.removeprefix(CUSTOM_TYPE_START_IDENTIFIER).removesuffix(
            CUSTOM_TYPE_END_IDENTIFIER
        )
        try:
            parts = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unparseable dspy custom-type marker: {e}") from e
        for part in parts if isinstance(parts, list) else [parts]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                if isinstance(url, str) and url:
                    return Image(url=url)
        raise ValueError("No image_url part found in dspy custom-type marker.")
    raise ValueError(f"Value is not a serialized dspy.Image: {type(serialized)}")


def rehydrate_audio_field(serialized: Any) -> Audio:
    """Rebuild a dspy.Audio from the marker string produced by model_dump.

    An Audio input field survives Signature.model_dump() only as a
    ``<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "input_audio", "input_audio":
    {"data": ..., "format": ...}}]<<CUSTOM-TYPE-END-IDENTIFIER>>`` string, and
    feeding that string back into a signature raises (the before-validator
    runs encode_audio on it and finds no usable source), so it must be
    converted to a real Audio object before the predictor call.

    Args:
        serialized: The dumped value of an Audio-typed field.

    Returns:
        Audio: The rehydrated audio object.

    Raises:
        ValueError: If the value is not a parseable dspy custom-type marker
            carrying an input_audio part.
    """
    if isinstance(serialized, Audio):
        return serialized
    if isinstance(serialized, str) and serialized.startswith(
        CUSTOM_TYPE_START_IDENTIFIER
    ):
        body = serialized.removeprefix(CUSTOM_TYPE_START_IDENTIFIER).removesuffix(
            CUSTOM_TYPE_END_IDENTIFIER
        )
        try:
            parts = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unparseable dspy custom-type marker: {e}") from e
        for part in parts if isinstance(parts, list) else [parts]:
            if isinstance(part, dict) and part.get("type") == "input_audio":
                payload = part.get("input_audio")
                if isinstance(payload, dict):
                    data = payload.get("data")
                    audio_format = payload.get("format")
                    if (
                        isinstance(data, str)
                        and data
                        and isinstance(audio_format, str)
                        and audio_format
                    ):
                        return Audio(data=data, audio_format=audio_format)
        raise ValueError("No input_audio part found in dspy custom-type marker.")
    raise ValueError(f"Value is not a serialized dspy.Audio: {type(serialized)}")


def rehydrate_video_field(serialized: Any) -> Video:
    """Rebuild a Video from the marker string produced by model_dump.

    A Video input field survives Signature.model_dump() only as a
    ``<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "file", "file": {"file_data":
    ...}}]<<CUSTOM-TYPE-END-IDENTIFIER>>`` string. Feeding that string back into
    a signature or predictor degrades the video to text garbage, so it must be
    converted to a real Video object before the predictor call.

    Args:
        serialized: The dumped value of a Video-typed field.

    Returns:
        Video: The rehydrated video object.

    Raises:
        ValueError: If the value is not a parseable dspy custom-type marker
            carrying a file part.
    """
    if isinstance(serialized, Video):
        return serialized
    if isinstance(serialized, str) and serialized.startswith(
        CUSTOM_TYPE_START_IDENTIFIER
    ):
        body = serialized.removeprefix(CUSTOM_TYPE_START_IDENTIFIER).removesuffix(
            CUSTOM_TYPE_END_IDENTIFIER
        )
        try:
            parts = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unparseable dspy custom-type marker: {e}") from e
        for part in parts if isinstance(parts, list) else [parts]:
            if isinstance(part, dict) and part.get("type") == "file":
                payload = part.get("file")
                if isinstance(payload, dict):
                    file_data = payload.get("file_data")
                    if isinstance(file_data, str) and file_data:
                        return Video(url=file_data)
        raise ValueError("No file part found in dspy custom-type marker.")
    raise ValueError(f"Value is not a serialized Video: {type(serialized)}")
