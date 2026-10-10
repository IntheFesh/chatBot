"""Serving the style model (round 14): local ``llama-server``, the SSH tunnel to a rented vLLM
instance, the checks before a model may be used, the evaluation that decides whether it may become
the default, and the fall-back when it breaks (R-SRV-001 to R-SRV-005).

The pieces, from the bottom up:

``hardware``        the graphics card (``nvidia-smi``) and which llama.cpp build fits it
``quant``           which quantisation of a model a card can hold
``llamacpp``        finding ``llama-server``, its command line, the managed process
``tunnel``          the ``asyncssh`` port forward to the instance, with reconnects
``tokencheck``      the tokenizer comparison that must pass before a model is used (R-TRN-011.4)
``activation``      ``twin model activate|disable`` and the audit trail of both
``gate_m5``         the judge of milestone M5 (R-SRV-005)
``evaluation``      ``twin model evaluate``: the same contexts through DeepSeek and the model
``component``       the application component that runs the server or the tunnel
``cli``             the commands
"""
