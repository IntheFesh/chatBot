# Test fixtures

Everything in this directory (and every value used by the tests) is **synthetic**.
Real chat records, media, databases, vector stores, training sets and model files must
never be committed (CLAUDE.md rule 6).  Fixture data that needs a pattern-shaped value
(a WeChat id, a phone number, ...) is built at run time by the generators in
`tests/support/`, so `scripts/privacy_scan.py` stays clean.
