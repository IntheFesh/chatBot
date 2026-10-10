"""The safety boundaries of the reply engine (R-SAFE-001 to R-SAFE-006).

========================  ==============================================================
:mod:`.crisis`            the user may be in danger: keyword screen, the model's second
                          judgement, the reply that steps out of the role, the alert
:mod:`.notifier`          the optional message to the emergency contact - only a time, no
                          chat content, in the one fixed template (``EmergencyNotifier``)
:mod:`.hotlines`          which help line belongs to the time zone the bot lives in
:mod:`.commitments`       promises of things the bot cannot do (calls, voice, photos, ...)
:mod:`.identity`          the user sincerely asks whether she is an AI
========================  ==============================================================

The post-processing (:mod:`twin.engine.postprocess`) uses :mod:`.commitments` and
:mod:`.identity`; the engine calls :class:`~twin.engine.safety.crisis.CrisisHandler` before it
generates anything.  The fixed rules in the prompt (``reply_rules`` template) state the same
boundaries to the model; the code here is the second line of defence.
"""
