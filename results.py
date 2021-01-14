class VesselDetectorResult:
    def __init__(
            self,
            id: str,
            area: float = None,
            solidity: float = None,
            max_width: int = None,
            max_height: int = None):
        self.id = id
        self.area = area
        self.solidity = solidity
        self.max_width = max_width
        self.max_height = max_height
