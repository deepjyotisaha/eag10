class HumanInterventionError(Exception):
    """Custom exception for human intervention related errors.
    
    This exception is raised when:
    1. Human intervention is disabled but attempted
    2. Timeout occurs while waiting for human input
    3. Human input is cancelled
    4. Any other error occurs during human intervention process
    """
    def __init__(self, message: str, error_type: str = "general"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

    def __str__(self):
        return f"HumanInterventionError ({self.error_type}): {self.message}"