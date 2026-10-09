import json
from typing import Any, Dict

from aap_core.types import ContentType, MediaReference, TokenUsage
from dspy import Image, Prediction
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
}


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
