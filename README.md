# aap_dspy

DSPy integration for the AI Agent Pattern (AAP) orchestration framework.

![PyPI - License](https://img.shields.io/pypi/l/aap_dspy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aap_dspy)
![PyPI - Version](https://img.shields.io/pypi/v/aap_dspy?label=%20)
![Python Version](https://img.shields.io/pypi/pyversions/aap_dspy)

## What is this?

`aap_dspy` provides a bridge between [DSPy](https://dspy.ai/) — the declarative programming framework for LMs — and the [AI Agent Pattern](https://github.com/quanghona/agent_design_pattern) orchestration layer (`aap_core`). It enables DSPy's signature-based prompt programming to work seamlessly within AAP's agent message flow, chain execution, and multi-agent orchestration patterns.

## Quick Start

```bash
pip install aap_dspy
```

```python
import dspy
from aap_dspy.chain import BaseSignatureAdapter, ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage

# 1. Define a DSPy Signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. Create a SignatureAdapter to convert between AgentMessage and dspy.Signature
class MyAdapter(BaseSignatureAdapter[QA]):
    def msg2sig(self, message: AgentMessage):
        # Convert AgentMessage -> List[QA]
        ...
    def sig2msg(self, signatures, name: str):
        # Convert List[QA] -> List[AgentResponse]
        ...

# 3. Build the chain
chain = ChatCausalMultiTurnsChain(signature=QA, adapter=MyAdapter())
chain = chain.with_lm(dspy.LM("openai/gpt-4o"))

# 4. Invoke within the AAP agent flow
message = AgentMessage(query="What is DSPy?")
result = chain.invoke(message)
```

## Relationship with `aap_core`

`aap_dspy` is a subpackage of the AAP monorepo that depends on `aap_core`. It does **not** replace the core orchestration layer — it extends it.

| Layer | Package | Role |
|-------|---------|------|
| Orchestration | `aap_core` | Agent lifecycle, multi-agent flows, message types, chain base classes |
| DSPy Integration | `aap_dspy` | Bridges DSPy signatures/predictors with `aap_core`'s `AgentMessage` flow |

The key integration point is the **`BaseSignatureAdapter`** abstract class. DSPy uses `dspy.Signature` to declare input/output fields and compose programs, while `aap_core` uses `AgentMessage` / `AgentResponse` for inter-agent communication. The adapter converts between the two:

- **`msg2sig`** — converts an `AgentMessage` into a list of `dspy.Signature` instances before they enter a DSPy predictor.
- **`sig2msg`** — converts the DSPy predictor output back into `AgentResponse` tuples for the agent flow.

This two-way conversion lets you use DSPy's declarative programming paradigm (compositional prompting, self-consistency, program optimization) inside any AAP orchestration pattern — sequential, parallel, voting, debate, etc.

## Features

- **`BaseSignatureAdapter`** — Abstract adapter class that converts between `AgentMessage` and `dspy.Signature`, with support for static prefill dictionaries and dynamic context injection.
- **`ChatCausalMultiTurnsChain`** — A multi-turn causal chain implementation that wraps DSPy predictors within `aap_core`'s `BaseCausalMultiTurnsChain` interface, supporting history tracking and intermediate step storage.
- **`RetrieverAdapter`** — Adapts DSPy's `Retrieve` and `Embeddings` retrievers to `aap_core`'s `BaseRetriever` interface for RAG workflows.
- **`token_from_response`** — Utility to extract token usage from DSPy `Prediction` objects into `aap_core`'s `TokenUsage` format.
- **Multi-LM support** — Each chain can be configured with a different LM via `with_lm()`, enabling mixed-model agent flows.

## Installation

### Core dependency

```bash
pip install aap_core
```

### DSPy integration

```bash
pip install aap_dspy
```

### Optional: Weaviate retriever backend

```bash
pip install aap_dspy[weaviate]
```

## Usage

### Defining a Signature and Adapter

```python
import dspy
from typing import Dict, List
from aap_dspy.chain import BaseSignatureAdapter
from aap_core.types import AgentMessage, AgentResponse

class Summarize(dspy.Signature):
    """Summarize the given text concisely."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

class SummarizeAdapter(BaseSignatureAdapter[Summarize]):
    def msg2sig(self, message: AgentMessage) -> List[Summarize]:
        return [Summarize(text=message.query)]

    def sig2msg(self, signatures: List[Summarize], name: str) -> List[AgentResponse]:
        return [(name, sig.summary) for sig in signatures]
```

### Using with a Chain

```python
from aap_dspy.chain import ChatCausalMultiTurnsChain
import dspy

chain = ChatCausalMultiTurnsChain(signature=Summarize, adapter=SummarizeAdapter())
chain = chain.with_lm(dspy.LM("openai/gpt-4o"))

message = AgentMessage(query="Long text to summarize...")
result = chain.invoke(message)
```

### Using a DSPy Retriever

```python
from aap_dspy.retriever import RetrieverAdapter
from dspy import Retrieve

retriever = RetrieverAdapter(retriever=Retrieve(k=5))
message = AgentMessage(query="Find docs about RAG")
result = retriever(message)
```

## Architecture

```
aap_core (orchestration)
    ├── AgentMessage / AgentResponse  ← inter-agent message types
    ├── BaseCausalMultiTurnsChain     ← chain execution interface
    ├── BaseRetriever                 ← retrieval interface
    └── BaseAgent                     ← agent lifecycle

aap_dspy (DSPy integration)
    ├── BaseSignatureAdapter          ← AgentMessage ↔ dspy.Signature bridge
    ├── ChatCausalMultiTurnsChain     ← DSPy predictor within causal chain
    ├── RetrieverAdapter              ← DSPy Retrieve/Embeddings → BaseRetriever
    └── token_from_response           ← DSPy Prediction → TokenUsage
```

## Examples

Full Jupyter notebook examples are available in the [example/dspy](../../example/dspy) directory:

| Example | Description |
|---------|-------------|
| [sequential.ipynb](../../example/dspy/sequential.ipynb) | Sequential multi-agent chain |
| [parallel.ipynb](../../example/dspy/parallel.ipynb) | Parallel agent execution |
| [voting.ipynb](../../example/dspy/voting.ipynb) | Majority-vote ensemble |
| [debate.ipynb](../../example/dspy/debate.ipynb) | Multi-agent debate pattern |
| [self_reflection.ipynb](../../example/dspy/self_reflection.ipynb) | Self-refinement loop |
| [cross_reflection.ipynb](../../example/dspy/cross_reflection.ipynb) | Cross-agent reflection |
| [iterative_refinement.ipynb](../../example/dspy/iterative_refinement.ipynb) | Iterative improvement |
| [coordinator.ipynb](../../example/dspy/coordinator.ipynb) | Coordinator-pattern agents |
| [retriever.ipynb](../../example/dspy/retriever.ipynb) | RAG with DSPy retrievers |
| [simple_loop.ipynb](../../example/dspy/simple_loop.ipynb) | Simple self-loop agent |

## Documentation

- Full project docs: [agent_design_pattern](https://github.com/quanghona/agent_design_pattern)
- DSPy documentation: [dspy.ai](https://dspy.ai)
- `aap_core` docs: [src/core/README.md](../core/README.md)

## Contributing

See the [main repository](../../README.md) for contribution guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](../../LICENSE) file for details.
