class Error(Exception):
    """
    Base exception for  custom error types.
    """


class NetworkError(Error):
    pass


class EnvironmentError(Error):
    pass


class UserError(Error):
    pass


class CheckpointError(Error):
    pass


class ConfigurationError(Error):
    pass


class CLIError(Error):
    pass


class ThreadError(Error):
    pass


class BeakerExperimentFailedError(Error):
    pass
