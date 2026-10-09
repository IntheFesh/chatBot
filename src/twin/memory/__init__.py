"""The memory: facts, daily summaries, the life line and follow-ups (round 07; R-MEM, R-TRN-013).

What later rounds import is in :mod:`twin.memory.api` (``Memory``, ``MemoryAssembler``,
``memory_view``) and :mod:`twin.memory.asof` (``AsOfView``).  By job:

* data: :mod:`~twin.memory.records` (decrypted rows), :mod:`~twin.memory.store` (the tables),
  :mod:`~twin.memory.corpus` (the snapshot with the keyword index, :mod:`~twin.memory.keywords`),
  :mod:`~twin.memory.vectors` (two vector tables), :mod:`~twin.memory.memory` (all of them
  together), :mod:`~twin.memory.localdate` (local days of her records and of the bot);
* what is visible when: :mod:`~twin.memory.visible` (the one definition), :mod:`~twin.memory.view`
  (``memory_view``), :mod:`~twin.memory.asof` (``AsOfView``);
* the memory block of a reply: :mod:`~twin.memory.assemble`, :mod:`~twin.memory.blocks`,
  :mod:`~twin.memory.render`;
* getting memories in: :mod:`~twin.memory.extract` (facts and follow-ups from a conversation),
  :mod:`~twin.memory.timeparse` (dates and times in Chinese), :mod:`~twin.memory.conflict` and
  :mod:`~twin.memory.writer` (the priority rules), :mod:`~twin.memory.summarize` (daily
  summaries), :mod:`~twin.memory.lifeline`, :mod:`~twin.memory.followups`;
* replay of the real history: :mod:`~twin.memory.replay`, :mod:`~twin.memory.hook` (after an
  import), :mod:`~twin.memory.jobs`;
* the user's commands: :mod:`~twin.memory.manage` (``remember`` / ``forget`` / ``list``),
  :mod:`~twin.memory.cli` (``twin memory``);
* the bot's conversation: :mod:`~twin.memory.recent` (``RecentTurns``, ``HistoryWindow``, the
  read interface that round 09 implements on ``bot_turns``).
"""
