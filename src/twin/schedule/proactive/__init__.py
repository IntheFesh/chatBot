"""Proactive messages: she writes first, at random, never while she sleeps (round 10).

Modules (none is imported here, so importing the package costs nothing):

* ``types``     - the trigger kinds, their priority, the candidates and the refusal reasons;
* ``rules``     - the hard constraints as one pure function, checked again right before sending;
* ``curve``     - the thinned Poisson process over the day's initiation-rate curve;
* ``slots``     - the fixed moments of a day (greeting, meals, goodnight) and the follow-up slot;
* ``store``     - ``proactive_candidates``, ``proactive_log`` and ``ratings`` over the database;
* ``settings``  - the runtime settings the next round writes (switch, daily range, pause);
* ``decide``    - the two-step decision: DeepSeek plans, the style model writes, the pipeline
  cleans;
* ``send``      - the sending at her pace, with the user able to interrupt;
* ``scheduler`` - the tick that ties them together;
* ``component`` - the process component that ticks on the aligned grid and listens for events;
* ``status``    - the proactive line of ``/状态``;
* ``cli``       - ``twin proactive log``.
"""
