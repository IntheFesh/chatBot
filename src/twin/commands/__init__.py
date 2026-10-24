"""In-chat (WeChat) commands (R-CMD-001 to R-CMD-003).

==========================  =========================================================
:mod:`.parse`               is this message a command?  (full-width slash, case and width of
                            names, colons and spaces between name and argument)
:mod:`.registry`            ``CommandSpec`` / ``CommandRegistry``: the table of commands - later
                            rounds register their commands here
:mod:`.router`              ``CommandRouter``: the engine's ``CommandPort`` implementation
:mod:`.builtin`             ``/帮助 /状态 /思考 /显示思考 /后端 /重来`` (round 09)
:mod:`.status`              the report of ``/状态`` and where its numbers come from
:mod:`.texts`               every word a command says, in one table
==========================  =========================================================

::

    from twin.commands.router import CommandRouter

    router = CommandRouter.from_services(services, llm, selector=style.selector, memory=memory)
    outcome = await router.handle(text, CommandContext(at, inbound_id))   # None: not a command
    router.register(CommandSpec(name="暂停", ...))                         # a later round's command
"""
