"""
SpeakTwin - Deep Learning Services
==================================
Optional neural models layered on top of the DSP pipeline.

Every module here follows the same contract:

  * it exposes an `is_enabled()` and an `analyze()`-style function
  * it returns `None` when the feature is off or its dependency is absent
  * it never raises into the request path

That contract is what lets `requirements.txt` stay lightweight: install
`requirements-ml.txt` to turn these on, and the backend behaves identically
either way apart from the extra fields in the response.

Note: the model registry is deliberately NOT re-exported here. Binding the
`ModelRegistry` instance as `backend.services.ml.registry` would shadow the
`backend.services.ml.registry` *module* of the same name, leaving the module
unreachable by dotted path. Import it explicitly instead:

    from backend.services.ml.registry import registry
"""

__all__: list[str] = []
