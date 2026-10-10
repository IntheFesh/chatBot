"""pytest plugin of the probe: no real Windows toast is shown."""

import twin.ops.notify as notify


def _no_toast(self: notify.WindowsToastNotifier, title: str, body: str) -> None:
    return None


notify.WindowsToastNotifier.notify = _no_toast  # type: ignore[method-assign]
