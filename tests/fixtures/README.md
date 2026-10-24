# Test fixtures

Everything in this directory (and every value used by the tests) is **synthetic**.
Real chat records, media, databases, vector stores, training sets and model files must
never be committed (CLAUDE.md rule 6).  Fixture data that needs a pattern-shaped value
(a WeChat id, a phone number, ...) is built at run time by the generators in
`tests/support/`, so `scripts/privacy_scan.py` stays clean.

`synth_export.py` writes a synthetic export with the structure of the real one (report, conversations
with `meta.json` / `messages.json`, media, optional `_integrity/`) for the import tests and for
`scripts/bench_import.py`.  Sentences are random strings of common characters, identifiers are
generated at run time, pictures are tiny generated images.

`tests/support/synth_chat.py` (round 04) writes conversations with *known regularities* straight
into `messages` (no export files): she is silent from 01:00 to 08:30, replies slowly on workdays
between 13:00 and 17:00, opens four conversations a day, uses a comma in 3 % of her texts;
`build_hourly_chat` follows an hour-of-day vector such as the 7-day sample of SPEC section 0.
The profile and routine tests assert that these truths are recovered within tolerance.
