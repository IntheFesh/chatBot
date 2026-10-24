"""Embedding model and the library of her real replies (round 05; R-RET, R-STO-005).

What later rounds import (nothing here loads the model or LanceDB at import time):

``twin.retrieval.embedder``
    ``embedding_service(services)`` - the shared :class:`EmbeddingService`; ``encode(texts, kind)``
    redacts (R-LLM-009), adds the model's query instruction for ``EncodeKind.QUERY`` and returns
    length-1 float32 vectors.  Memory (round 07) and sticker descriptions (round 06) use it.
``twin.retrieval.vector_store``
    ``VectorStore(services.paths.vectors_dir).table(schema)`` - one LanceDB table per kind of
    data (``GENERIC_SCHEMA`` of :mod:`twin.storage.vector_schema` for memory and stickers,
    ``WINDOW_SCHEMA`` for the example windows): ``upsert``, ``search``, ``delete_ids``, ``drop``
    and the ``<table>.meta.json`` that ties a table to one encoding.
``twin.retrieval.query``
    ``ExampleRetriever(services).query(ExampleQuery(turns, local_minute, day_type, before, k))``
    returns :class:`~twin.retrieval.examples.Example` objects; ``before=`` is the as-of bound of
    the evaluation sandbox and ``AsOfView(t)`` (R-TRN-013).
``twin.retrieval.examples``
    ``render_example()`` / ``render_examples()`` - the one place an example becomes prompt text.
``twin.retrieval.windows`` / ``twin.retrieval.records``
    how windows are laid out and the guards that keep everything but her real ``messages`` rows out.
``twin.retrieval.indexer``
    ``run_index()`` (job ``retrieval_index``), ``collect_stats()``; the CLI is ``twin retrieval``.
"""
