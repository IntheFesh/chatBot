"""Facts about the DeepSeek API taken from the official documentation.

Verified on 2026-10-09 against these pages (English; the Chinese pages agree):

* models and prices - https://api-docs.deepseek.com/quick_start/pricing
* thinking mode     - https://api-docs.deepseek.com/guides/thinking_mode
* context caching   - https://api-docs.deepseek.com/guides/kv_cache
* vision            - https://api-docs.deepseek.com/guides/vision
* JSON output       - https://api-docs.deepseek.com/guides/json_mode
* error codes       - https://api-docs.deepseek.com/quick_start/error_codes
* rate limits       - https://api-docs.deepseek.com/quick_start/rate_limit

What the pages say, as used by this package:

* Model names ``deepseek-flash`` (vision, thinking and non-thinking) and ``deepseek-v4-pro``
  (no vision).  The older names ``deepseek-v4-flash`` and ``deepseek-v4-flash-vision-exp`` are
  still accepted and billed at the Flash price.  Context 1M tokens, output up to 384K.
* Thinking is **on by default**; it is switched with ``{"thinking": {"type": "enabled" |
  "disabled"}}`` in the request body (``extra_body`` with the OpenAI SDK) and the effort with
  ``reasoning_effort`` (``low`` | ``high`` | ``max``).  ``temperature``, ``presence_penalty``
  and ``frequency_penalty`` are accepted but have no effect in thinking mode.  Without a
  ``tools`` parameter ``reasoning_content`` of earlier turns is ignored by the API.
* Prices (USD per million tokens) are listed for peak and off-peak; off-peak is half of peak.
  Peak is Monday to Friday excluding Chinese public holidays, 01:00-04:00 and 06:00-10:00 UTC
  (09:00-12:00 and 14:00-18:00 Beijing time).  Weekends and public holidays are off-peak "in
  full".  The pages do not mention compensatory working days on a weekend; see DECISIONS.md.
* Usage carries ``prompt_cache_hit_tokens`` and ``prompt_cache_miss_tokens``.  A cache hit needs
  a previously persisted identical prefix; a request writes prefix units at the end of its user
  input and of its output, building takes seconds, and the cache is best effort.
* Vision: JPEG, PNG, GIF and WebP, detected from the content; ``detail`` is ``low`` (downscaled
  to 512x512), ``high``, ``original`` or ``auto``; images are accepted in user messages only
  (system or assistant messages answer 400); at most 1024 tokens per image; 48 MiB request body,
  32 MiB per inline image, 600 images per request, 8192 px per side (4096 px with 15 or more
  images).
* JSON output: ``response_format={"type": "json_object"}``, the word "json" must appear in the
  prompt, ``max_tokens`` must leave room, and the API may occasionally return empty content.
* Errors: 400 and 422 invalid request, 401 authentication, 402 insufficient balance, 429 rate
  limit, 500 server error, 503 overloaded.  The server holds the connection open (empty lines)
  while a request waits and closes it after 10 minutes without inference.
"""

from __future__ import annotations

from datetime import date
from types import MappingProxyType

CHECKED_ON = date(2026, 10, 9)
DOC_PAGES = MappingProxyType(
    {
        "pricing": "https://api-docs.deepseek.com/quick_start/pricing",
        "thinking_mode": "https://api-docs.deepseek.com/guides/thinking_mode",
        "kv_cache": "https://api-docs.deepseek.com/guides/kv_cache",
        "vision": "https://api-docs.deepseek.com/guides/vision",
        "json_output": "https://api-docs.deepseek.com/guides/json_mode",
        "error_codes": "https://api-docs.deepseek.com/quick_start/error_codes",
        "rate_limit": "https://api-docs.deepseek.com/quick_start/rate_limit",
    }
)

BASE_URL = "https://api.deepseek.com"
FLASH_MODEL = "deepseek-flash"
PRO_MODEL = "deepseek-v4-pro"
LEGACY_MODEL_ALIASES = MappingProxyType(
    {
        "deepseek-v4-flash": FLASH_MODEL,
        "deepseek-v4-flash-vision-exp": FLASH_MODEL,
    }
)
VISION_MODELS = frozenset({FLASH_MODEL})

# USD per million tokens at peak hours; off-peak is OFFPEAK_RATIO times these.
PEAK_PRICES_USD_PER_MTOK = MappingProxyType(
    {
        FLASH_MODEL: MappingProxyType({"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20}),
        PRO_MODEL: MappingProxyType({"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96}),
    }
)
OFFPEAK_PRICES_USD_PER_MTOK = MappingProxyType(
    {
        FLASH_MODEL: MappingProxyType({"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.60}),
        PRO_MODEL: MappingProxyType({"cache_hit": 0.022, "cache_miss": 0.66, "output": 1.98}),
    }
)
OFFPEAK_RATIO = 0.5
# [start, end) hours in UTC on a Beijing working day
PEAK_HOURS_UTC = ((1, 4), (6, 10))
BEIJING_UTC_OFFSET_HOURS = 8

REASONING_EFFORTS = ("low", "high", "max")
DETAIL_VALUES = ("low", "high", "original", "auto")
IMAGE_FORMATS = ("image/jpeg", "image/png", "image/gif", "image/webp")
MAX_IMAGE_TOKENS = 1024
MAX_IMAGES_PER_REQUEST = 600
MAX_IMAGE_SIDE_PX = 8192
MAX_IMAGE_SIDE_PX_MANY = 4096
MANY_IMAGES_THRESHOLD = 15
MAX_INLINE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024
LOW_DETAIL_SIDE_PX = 512

# rough size of the characters-to-tokens ratio quoted on the token page (tokens per character)
TOKENS_PER_CJK_CHAR = 0.6
TOKENS_PER_LATIN_CHAR = 0.3
