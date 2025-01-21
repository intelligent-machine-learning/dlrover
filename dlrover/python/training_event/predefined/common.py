from ..emitter import Process


class CommonEventName:
    WARNING = "#warning"
    EXCEPTION = "#exception"


class WarningType:
    DEPRECATED = "deprecated"


class CommonPredefined(Process):
    """
    Common predefined events.
    """

    def __init__(self, target: str):
        super().__init__(target)

    def warning(self, warning_type: WarningType, msg: str, **kwargs):
        """
        Emit a warning event, the warning event will be notified to the user.
        """
        self.instant(
            CommonEventName.WARNING,
            {
                "warning_type": warning_type,
                "msg": msg,
                **kwargs,
            },
        )
