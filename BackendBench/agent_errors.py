class AgentError(Exception):
    """
    Exception raised for errors related to LLM/agent failures,
    such as rate limits, empty code, bad formatting, or API issues.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message