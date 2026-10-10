"""pytest plugin of the probe: the power monitor never creates its hidden window."""

import twin.ops.power_events as power_events


def _no_window(self: power_events.PowerEventMonitor) -> None:
    return None


power_events.PowerEventMonitor._start_window = _no_window  # type: ignore[method-assign]
