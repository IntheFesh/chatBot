"""The post-processing of a generated reply (R-ENG-008, R-ENG-012, R-STK-005, R-SAFE-002/006).

::

    from twin.engine.postprocess import PostContext, PostProcessor, fit_bubbles_to_quota

    result = PostProcessor().process(raw_output, context)     # bubbles, actions, violations
    bubbles, actions = fit_bubbles_to_quota(result.bubbles, remaining)   # at the moment of sending

The steps and their order are documented in :mod:`.steps`; :func:`fit_bubbles_to_quota` is the
platform-quota step on its own, for the engine to call with the quota left at the moment it sends.
"""

from twin.engine.postprocess.model import PostContext, Working
from twin.engine.postprocess.phrases import AiPhrases, load_ai_phrases
from twin.engine.postprocess.processor import PostProcessor, PostResult
from twin.engine.postprocess.quota import fit_bubbles_to_quota
from twin.engine.postprocess.steps import STEPS
from twin.engine.postprocess.style import StyleLimits

__all__ = [
    "STEPS",
    "AiPhrases",
    "PostContext",
    "PostProcessor",
    "PostResult",
    "StyleLimits",
    "Working",
    "fit_bubbles_to_quota",
    "load_ai_phrases",
]
