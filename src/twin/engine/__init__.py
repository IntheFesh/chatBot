"""The reply engine (round 09): from a message of the user to the bubbles she would send.

Step 1 (this part of the package, stateless):

==========================  =========================================================
:mod:`.pipeline`            ``ReplyPipeline.run(context, data_view) -> ReplyDraft`` - the one
                            place where a reply is written, checked and post-processed
:mod:`.dataview`            ``ReplyDataView`` and the live view (``AsOfView`` fits it too)
:mod:`.prompt`              the prompt of the DeepSeek backend, laid out for the cache
:mod:`.backend`,            ``ReplyBackend`` and the DeepSeek backend
:mod:`.deepseek_backend`
:mod:`.parsing`,            the output convention and the post-processing steps
:mod:`.postprocess`
:mod:`.safety`              crisis, promises, "are you an AI?", the emergency contact
:mod:`.inbound`             how each kind of message of the user reads in the prompt
:mod:`.turns`,              ``bot_turns``, ``conversation_state``, ``feedback`` and the window
:mod:`.state_store`,        of the recent conversation
:mod:`.feedback`,
:mod:`.history`
==========================  =========================================================

:mod:`twin.engine.api` lists what other rounds import.
"""
