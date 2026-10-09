"""The reply engine (round 09): from a message of the user to the bubbles she would send.

Step 1 (stateless):

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

Step 2 (the style model's side and the commands' port):

==========================  =========================================================
:mod:`.style_prompt`        ``StylePromptBuilder``: the style model's prompt, for the running
                            bot and for the training export (ChatML of ``qwen3_nothink``,
                            ``normalize_context``, token budget, locked versions)
:mod:`.style_backend`,      the ``style`` and ``hybrid`` backends (DeepSeek plans, the style
:mod:`.hybrid_backend`      model writes)
:mod:`.style_models`        the active model of ``model_registry`` and its locked versions
:mod:`.backend_select`      which backend answers: the setting, the fallback to DeepSeek when
                            the style model fails, the way back (R-SRV-004)
:mod:`.style_runtime`       client, backends and selector wired from the services
:mod:`.command_port`        what the engine asks of the command router (``twin.commands``)
==========================  =========================================================

:mod:`twin.engine.api` lists what other rounds import.
"""
