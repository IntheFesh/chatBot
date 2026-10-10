"""Stickers and WeChat emoji codes (round 06; R-STK-001 to R-STK-006), and the sticker library.

What later rounds import:

``twin.stickers.selector``
    ``StickerSelector(services, view=...).choose(tag, context_text, recent_bubbles)`` - one of her
    stickers for a tag (``as_of_view(t)`` for the training export and the evaluation sandbox);
``twin.stickers.rate``
    ``StickerRateController.from_services(services).should_drop()`` / ``record(is_sticker)`` - keeps
    the share of stickers near hers;
``twin.stickers.emoji_codes``
    ``EmojiCodePolicy.from_profile(services, scope)`` - her emoji-code vocabulary, rate and runs;
``twin.stickers.incoming``
    ``describe_incoming_sticker(services, md5=..., image=...)`` - what the user's sticker shows;
``twin.stickers.catalog`` / ``twin.stickers.tags``
    the library with its tags, the closed tag vocabulary and how tags from the picture, from her
    use and by hand are combined.

The rest: ``tagging`` (the vision model and the context correction), ``tag_jobs`` (planning and
running them, with the one-time batch of R-LLM-014), ``vectors`` (description vectors in a table
of their own), ``hook`` (after an import and after a re-split), ``download`` and ``library`` (round
03: files, status and the outbound allow-list), ``cli`` (``twin stickers``).
"""
