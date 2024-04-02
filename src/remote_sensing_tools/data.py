"""Data loading"""

from dataclasses import dataclass


@dataclass
class LoadInfo:
    """
    Information required for loading
    """

    bbox: tuple
    crs: str
    bands: list
    resolution: int
    start_date: str
    end_date: str


class EOData:
    """EOData class"""

    def __init__(self, load_info: LoadInfo) -> None:
        self.load_info = load_info
