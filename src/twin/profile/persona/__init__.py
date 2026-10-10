"""The persona card (round 06; R-PERS-001 to R-PERS-005).

What later rounds import is in :mod:`twin.profile.persona.api` (``load_persona``,
``render_full``, ``render_compact``, ``read_corrections``, ``write_corrections``).  The rest, by
job:

* the text of a card: :mod:`~twin.profile.persona.sections` (the four sections, cut without
  changing a byte, and the priority tier of every line), :mod:`~twin.profile.persona.compose`
  (building a card from its parts), :mod:`~twin.profile.persona.render` (the two renderings with
  their token budgets);
* versions: :mod:`~twin.profile.persona.store` (``persona_cards`` and the version in force of
  each scope), :mod:`~twin.profile.persona.refresh` (fresh statistics, the pre-holdout copy of
  the hand-written style lines, when the description is due, writing a new description and the
  corrections section);
* the description from real conversations: :mod:`~twin.profile.persona.sampling` (stratified
  choice of segments), :mod:`~twin.profile.persona.generate` (Map-Reduce with checked evidence),
  :mod:`~twin.profile.persona.jobs` (the one-time batch and the refresh job);
* people: :mod:`~twin.profile.persona.edit` (the editor session of ``twin persona edit``),
  :mod:`~twin.profile.persona.cli`, and :mod:`~twin.profile.persona.hook` (after an import and
  after a re-split of the hold-out).

Two scopes exist for everything (R-TRN-013): ``live`` (all data, for the running bot) and
``pre_holdout`` (data before ``holdout_cutoff()``, for training and evaluation).  The prompt
templates the description is made with are in :mod:`twin.profile.prompt_templates`.
"""
