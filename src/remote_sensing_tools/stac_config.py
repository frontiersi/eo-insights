"""Construct a STAC config object"""

from dataclasses import dataclass
from typing import Union, Optional
import pathlib
import tomli

PathType = Union[str, pathlib.Path]


@dataclass
class CatalogInfo:
    """Data class for a STAC Catalog"""

    name: str
    url: str
    rio_config: dict


@dataclass
class CollectionInfo:
    """Data class for a STAC Collection"""

    id: str
    description: str
    aliases: dict
    assets: dict
    masks: dict


class STACConfig:
    """STAC config class"""

    def __init__(
        self,
        config_file_path: PathType,
        configuration: Optional[dict],
        catalog: Optional[CatalogInfo],
        collections: Optional[dict[str, CollectionInfo]],
    ) -> None:
        self.config_file_path = config_file_path
        self.configuration = configuration
        self.catalog = catalog
        self.collections = collections

        if self.config_file_path is not None:
            # Load in the whole configuration dictionary
            self.configuration = self.config_dictionary_from_toml(self.config_file_path)

            # Set up attributes for STAC Catalog settings
            catalog_dict = self.configuration.get("catalog", {})
            self.catalog = CatalogInfo(
                name=catalog_dict.get("name", ""),
                url=catalog_dict.get("url", ""),
                rio_config=catalog_dict.get("rio_config", {}),
            )

            # Set up attributes for collections
            collections_settings = self.configuration.get("collections", {})

            self.collections = {}
            for collection, settings in collections_settings.items():

                self.collections[collection] = CollectionInfo(
                    id=collection,
                    description=settings.get("description", ""),
                    aliases=settings.get("aliases", {}),
                    assets=settings.get("assets", {}),
                    masks=settings.get("masks", {}),
                )

    def __str__(self) -> str:
        return f"Configuration constructed from {self.config_file_path}"

    def __repr__(self) -> str:
        return f"STACConfig('{self.config_file_path}, {self.configuration}, {self.catalog}, {self.collections}')"

    def config_dictionary_from_toml(self, config_file_path) -> dict:
        """Load the configuration dictionary from the TOML file"""
        with open(config_file_path, mode="rb") as f:
            config = tomli.load(f)

        return config
