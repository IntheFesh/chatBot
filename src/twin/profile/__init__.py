"""Statistical style profile, activity (routine) model and the hold-out split point.

What later rounds import is in :mod:`twin.profile.api` (``holdout_cutoff``, ``load_profile``,
``load_activity_model``).  The rest, by job:

* units and text: :mod:`~twin.profile.units` (bursts, segments, latencies, initiations),
  :mod:`~twin.profile.textstats` (punctuation, emoji, particles), :mod:`~twin.profile.localtime`
  (the local clock of the place she was in);
* the profile: :mod:`~twin.profile.metrics` (one pass of counters per window),
  :mod:`~twin.profile.values` and :mod:`~twin.profile.snapshot` (the stored JSON and blending),
  :mod:`~twin.profile.distribution` (sampleable empirical distributions),
  :mod:`~twin.profile.rules` (numeric style rules), :mod:`~twin.profile.phrases` (sealed text
  statistics), :mod:`~twin.profile.diffing` (version differences);
* the routine: :mod:`~twin.profile.activity_collect`, :mod:`~twin.profile.activity_infer`,
  :mod:`~twin.profile.activity_model` (queries and ``typical_state``),
  :mod:`~twin.profile.circular`, :mod:`~twin.profile.overrides` (manual corrections);
* versions and scopes: :mod:`~twin.profile.builder` (recomputation), :mod:`~twin.profile.store`
  (history, active version, rollback), :mod:`~twin.profile.holdout` (the one split point);
* wiring: :mod:`~twin.profile.hook`, :mod:`~twin.profile.jobs`, :mod:`~twin.profile.queue`,
  :mod:`~twin.profile.report_section`, :mod:`~twin.profile.cli`, :mod:`~twin.profile.show`.

Two scopes exist for everything derived from the data (R-TRN-013): ``live`` (all messages, for
the running bot) and ``pre_holdout`` (messages before ``holdout_cutoff()``, for training and
evaluation).
"""
