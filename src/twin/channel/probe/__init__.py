"""The M0 channel probe (R-CH-009, R-CH-010): what the WeChat ClawBot channel really allows.

Modules, from the bottom up:

``model``     the persisted plan: steps, attempts, actions, questions (pure data)
``summary``   the structured result stored for the milestone check, and the R-CH-010 verdict
``images``    the synthetic test pictures (JPEG, PNG, animated GIF), drawn with Pillow
``policy``    ``ProbeSendPolicy``: the only object allowed to skip the safe thresholds
``store``     the plan and the audit trail in the encrypted ``channel_state`` table
``steps``     what each measurement sends and how its outcome is judged
``runner``    the state machine that carries the plan out over hours, restart-safe
``component`` ``ChannelProbe``, the application component that runs the state machine
``report``    ``docs/CHANNEL_REPORT.md``
``cli``       ``twin channel probe start | status | answer | stop | report``

This ``__init__`` imports nothing so that light modules (``summary``) can be used on their
own, for example by the channel factory and by the milestone check.
"""
