import abc
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union, get_args, get_origin
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, AgentResponse, ContentType, TokenUsage
from aap_core.utils import extract_repo_id, huggingface_capabilities
from .utils import (
    model_info_to_capabilities,
    rehydrate_image_field,
    token_from_response,
)
import dspy
import litellm
from pydantic import Field, PrivateAttr

# Modalities the media gate can consult; detection reports on these.
_GATED_MODALITIES: Tuple[ContentType, ...] = ("image", "audio", "video", "document")


# litellm's provider names, used to tell "provider/repo-id" model strings from
# repo ids that merely contain a slash. The hf aliases are not enum values but
# are documented dspy/litellm shorthands for the Hugging Face Inference API.
def _provider_prefixes() -> frozenset:
    try:
        names = {str(value.value).lower() for value in litellm.LlmProviders}
    except Exception:  # noqa: BLE001 - tolerate litellm layout changes
        names = set()
    return frozenset(names) | {"hf", "hugging_face"}


_LITELLM_PROVIDER_PREFIXES = _provider_prefixes()


Signature = TypeVar("Signature", bound=dspy.Signature)


class BaseSignatureAdapter(abc.ABC, Generic[Signature]):
    """The adapter convert between AgentMessage and dspy.Signature
    In this class we also have the prefill dictionary to fill in values to the Signature fields.
    This is useful when we have static fields that don't exist in the AgentMessage object while it is moving in the workflow"""

    _prefill_dict: Dict[str, Any] = PrivateAttr({})

    @abc.abstractmethod
    def msg2sig(self, message: AgentMessage) -> List[Signature]:
        """The signature fields are only known when developing end application.
        This function convert AgentMessage fields to dspy Signature before flow into the dspy predictor.
        The filling logic for signature should be implemented in this method in the child class

        Specifically, there are 2 attributes need to taken care of:
        - prefill dictionary in this adapter class. This is also known as the static filling
        - the context dictionary in the AgentMessage. This is also known as the dynamic filling

        Args:
            message (AgentMessage): message to convert

        Returns:
            List[Signature]: list of the conversation so fat in dspy.Signature format"""
        raise NotImplementedError

    @abc.abstractmethod
    def sig2msg(self, signatures: List[Signature], name: str) -> List[AgentResponse]:
        """dspy.Signature to AgentMessage mapping.
          This function used after the flow is completed and the dspy output need to convert back to the agent message.

          Note about extracting the source name who generate the message and the message content from signature.
          The source can be assistant or tool, the user message is the dspy.InputField, and dspy already handled the system message.
          To unify about the source name, we can make the following assumptions:
          If a signature have both OutputField and ToolCalls, it is a tool message. Otherwise it is an assistant message

        Args:
            signatures (List[Signature]): dspy output
            name (str): agent name

        Returns:
            List[AgentResponse]: list of responses extract from the signatures input
        """
        raise NotImplementedError

    # Media (image input) adapter pattern:
    # dspy delivers media through signature fields typed with the native
    # dspy.Image type, not through message content parts. An adapter that
    # supports image input declares the field on its signature and maps
    # AgentMessage.query_media through the chain's gated funnel:
    #
    #     class VisionSig(dspy.Signature):
    #         image: Optional[dspy.Image] = dspy.InputField(default=None)
    #         question: str = dspy.InputField()
    #         answer: str = dspy.OutputField()
    #
    #     class MyAdapter(BaseSignatureAdapter[VisionSig]):
    #         def __init__(self):
    #             self.chain = None  # set right after chain construction
    #
    #         def msg2sig(self, message):
    #             images = self.chain._image_parts(message)  # gate applied here
    #             return [
    #                 VisionSig(
    #                     question=message.query,
    #                     image=(media_ref_to_image(images[0]) if images else None),
    #                     answer="",
    #                 )
    #             ]
    #
    # Two dspy constraints shape this pattern:
    # - Signature instances validate every field, so output fields need
    #   placeholders (answer="") and the image field must be
    #   Optional[dspy.Image] with default=None to carry "no media" on text-only
    #   turns; model_dump(exclude_none=True) in _generate_response then drops
    #   the None image before the predictor call.
    # - dspy emits native image parts only for dspy.Image *objects*; a plain
    #   URL string would be interpolated as text. Hence the chain rehydrates
    #   Image-typed fields around the model_dump round-trip.
    #
    # Always obtain refs via chain._image_parts / chain._media_parts, never via
    # resolve_media directly: those are the only paths where the per-modality
    # capability gate (media_support / detect_capabilities) is applied, so a
    # non-vision model drops media instead of failing at the API. The chain
    # also rehydrates Image fields around the model_dump round-trip in
    # _generate_response, so dumped conversations keep working.

    @classmethod
    def with_prefill(cls, prefill_dict: Dict[str, str]) -> "BaseSignatureAdapter":
        """Create a new instance of the adapter with the given prefill dictionary.
        Note that the child class is responsible for manage the matching and evaluation between prefill dictionary and signature

        Args:
            prefill_dict (Dict[str, str]): The prefill dictionary.

        Returns:
            BaseSignatureAdapter: A new instance of the adapter with the given prefill dictionary.
        """
        obj = cls()
        obj._prefill_dict = prefill_dict
        return obj

    def add_prefill(self, key: str, value: Any) -> None:
        """Add a new key-value pair to the prefill dictionary.

        Args:
            key (str): The key to add.
            value (str): The value to add.
        """
        self._prefill_dict[key] = value

    def remove_prefill(self, key: str) -> None:
        """Remove a key from the prefill dictionary.

        Args:
            key (str): The key to remove.
        """
        del self._prefill_dict[key]


class ChatCausalMultiTurnsChain(
    BaseCausalMultiTurnsChain[dspy.Signature, dspy.Prediction],
    arbitrary_types_allowed=True,
):
    """A class that handle LM call using dspy without history.

    Regarding tool calling pattern used in dspy framework, there are 2 approaches proposed by authors of dspy:
    1. [dspy fully managed](https://dspy.ai/learn/programming/tools/#approach-1-using-dspyreact-fully-managed): using dspy.ReAct or its subclass or customized dspy.Module that handle tool calling internally.
    In this case, first the signature of module doesn't have dspy.ToolCalls field. All tools completely stay inside the dspy module.
    This class only get the final output produced by dspy predictor. The _process_tools will not be call at all.

    2. [Manual tool handling](https://dspy.ai/learn/programming/tools/#approach-1-using-dspyreact-fully-managed): tool calling logic is handled by this class.
    When initializing this class with provided signature, this class will automaticallty detect the dspy.ToolCalls field.
    When invoke the chain, it will detects and calls the tool depends on the value of the tool calls field.

    Reference: https://dspy.ai
    """

    predictor: dspy.Module = Field(..., description="dspy predictor")
    adapter: BaseSignatureAdapter = Field(
        ...,
        description="The adapter convert between AgentMessage and dspy.Signature",
    )
    _signature: type[dspy.Signature] = PrivateAttr()
    _tool_calls_field: str | None = PrivateAttr(None)
    _lm: dspy.LM | None = PrivateAttr(None)
    _history_field_name: str | None = PrivateAttr(None)
    _capabilities_cache: Dict[ContentType, bool] | None = PrivateAttr(default=None)
    _capabilities_lm: Any = PrivateAttr(default=None)

    def __init__(self, signature: str | type[dspy.Signature], **kwargs):
        super().__init__(**kwargs)
        self._signature = dspy.ensure_signature(signature)
        for key, value in self._signature.input_fields.items():
            if value.annotation is dspy.History:
                self._history_field_name = key
                break
        for key, value in self._signature.output_fields.items():
            if value.annotation is dspy.ToolCalls:
                self._tool_calls_field = key
                break

    def detect_capabilities(self) -> Dict[ContentType, bool]:
        """Resolve per-modality input support from model metadata, not model names.

        Sources, in priority order:
        1. What the user declared in media_support ("enabled"/"disabled"). The
           core gate applies that before this method runs, so an explicit
           declaration always wins.
        2. The litellm model-info table (litellm.get_model_info): dspy has no
           capability metadata of its own - dspy.LM takes "provider/model"
           strings "supported by LiteLLM" and delegates everything to it, so
           litellm's provider-maintained flags are the authoritative source.
        3. Hugging Face model-card metadata, for models identified by repo id
           (needs the "hf" extra).

        Model-family names carry no weight here: variants of one family differ
        per modality (an audio-capable small model and a larger sibling without
        audio input share a name prefix), so name matching misreports exactly
        the cases that matter. Modalities no source reports on stay absent, and
        auto mode treats them as capable (fail loud).

        The result is cached per LM object and recomputed when the LM is
        replaced, since sources 2 and 3 can cost a network request.
        """
        lm = self._active_lm()
        if self._capabilities_cache is None or self._capabilities_lm is not lm:
            self._capabilities_cache = self._resolve_capabilities(lm)
            self._capabilities_lm = lm
        return dict(self._capabilities_cache)

    def _active_lm(self) -> Any:
        """The most specific LM in scope: with_lm(), then predictor-level, then global."""
        return self._lm or getattr(self.predictor, "lm", None) or dspy.settings.lm

    def _resolve_capabilities(self, lm: Any) -> Dict[ContentType, bool]:
        """Query the metadata sources once, filling gaps from the next source."""
        capabilities = self._litellm_capabilities(lm)
        if set(capabilities) != set(_GATED_MODALITIES):
            repo_id = extract_repo_id(self._identifier_candidates(lm))
            if repo_id is not None:
                for modality, value in huggingface_capabilities(repo_id).items():
                    capabilities.setdefault(modality, value)
        return capabilities

    def _litellm_capabilities(self, lm: Any) -> Dict[ContentType, bool]:
        """Read the input-modality flags from litellm's model-info table.

        get_model_info consults a static map for most providers but can reach
        the network for others (e.g. ollama queries the server). Any failure -
        unmapped revision, offline server, missing credentials - reports
        undecidable rather than breaking the request path.
        """
        name = getattr(lm, "model", None)
        if not isinstance(name, str) or not name:
            return {}
        try:
            info = litellm.get_model_info(model=name)
        except Exception:  # noqa: BLE001 - detection must never break invoke()
            return {}
        return model_info_to_capabilities(info)

    def _identifier_candidates(self, lm: Any) -> List[str]:
        """Model identifiers to test for a Hugging Face repo id, best first.

        dspy model strings are "provider/name", and a bare "provider/name"
        would otherwise pass for an org/repo pair ("ollama_chat/gemma3"
        included), so the prefix is stripped only when it names a litellm
        provider; for those, the remainder is the candidate - for providers
        hosting HF weights (hosted_vllm, huggingface, ...) it is the repo id.
        Unprefixed or unknown-prefix strings are offered whole. Plain model
        names ("gpt-4o", "gemma3") match no repo-id form and stay undecidable
        instead of being looked up and 404'd.
        """
        name = getattr(lm, "model", None)
        if not isinstance(name, str) or not name:
            return []
        head, sep, rest = name.partition("/")
        if sep and head.lower() in _LITELLM_PROVIDER_PREFIXES:
            return [rest]
        return [name]

    def _prepare_conversation(self, message: AgentMessage) -> List[dspy.Signature]:
        return self.adapter.msg2sig(message)

    def _generate_response(
        self, conversation: List[dspy.Signature], **kwargs
    ) -> Tuple[List[dspy.Signature], dspy.Prediction, bool, TokenUsage]:
        sig = conversation[-1].model_dump(exclude_none=True)
        # model_dump serializes dspy.Image fields to custom-type marker strings,
        # which the predictor would treat as plain text. Rehydrate them so the
        # adapter can emit native image content parts.
        self._rehydrate_media(sig)
        if self._history_field_name is not None:
            # Convert dict to object as the library use object to access the history
            history = dspy.History(messages=sig[self._history_field_name]["messages"])
            sig[self._history_field_name] = history
        if self._lm:
            # change context if possible
            with dspy.context(lm=self._lm, track_usage=True):
                data = self.predictor(**sig)
                usage = token_from_response(data)
        else:
            data = self.predictor(**sig)
            usage = token_from_response(data)

        has_tool = (
            False
            if self._tool_calls_field is None
            else bool(data[self._tool_calls_field])
        )
        sig.update(data.items())
        # The prediction echoes input fields, Image fields included, as marker
        # strings again; rehydrate so the appended conversation entry stays
        # valid for the next round-trip.
        self._rehydrate_media(sig)
        conversation.append(self._signature(**sig))
        return conversation, data, has_tool, usage

    def _rehydrate_media(self, sig: Dict[str, Any]) -> None:
        """Convert serialized dspy.Image input fields in a kwargs dict back to Image.

        In-place. A field counts as image-typed when its annotation is
        dspy.Image or Optional[dspy.Image] (the pattern adapters use to carry
        None on text-only turns). Values that are already Image objects, or
        absent from the dict, are left alone.
        """
        for key, field in self._signature.input_fields.items():
            annotation = field.annotation
            image_typed = annotation is dspy.Image or (
                get_origin(annotation) is Union and dspy.Image in get_args(annotation)
            )
            if image_typed and isinstance(sig.get(key), str):
                sig[key] = rehydrate_image_field(sig[key])

    def _process_tools(
        self, conversation: List[dspy.Signature], response: dspy.Prediction
    ) -> List[dspy.Signature]:
        for call in response[self._tool_calls_field].tool_calls:
            result = call.execute()
            for key, value in self._signature.output_fields.items():
                if value.annotation is str:
                    sig = self._signature(**response, **{key: result})
                    conversation.append(sig)
                    break

        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[dspy.Signature]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 1
            if self.store_immediate_steps
            else len(conversation) - 1
        )
        end_index = len(conversation)
        message.responses.extend(
            self.adapter.sig2msg(conversation[start_index:end_index], self.name)
        )
        # TODO: handle other modals later
        return message

    def with_lm(self, lm: dspy.LM | None) -> "ChatCausalMultiTurnsChain":
        """Set the language model context to use for the chain.
        If set to None, the default context will be used.

        Args:
            lm (dspy.LM | None): The language model context to use for the chain.

        Returns:
            ChatCausalMultiTurnsChain: The updated ChatCausalMultiTurnsChain object."""
        self._lm = lm
        return self
