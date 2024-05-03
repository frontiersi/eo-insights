"""Construct a STAC config object"""

import logging

from dataclasses import dataclass
from typing import Union, Any
import pathlib
import tomli

PathType = Union[str, pathlib.Path]

# Set up the logger
_log = logging.getLogger(__name__)


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


@dataclass
class MaskInfo:
    """Data class for masking information from STAC config"""

    id: str
    alias: str
    description: str
    collection: str
    type: str
    categories_to_mask: list[str]
    flags_definition: dict[str, Any]


class STACConfig:
    """STAC config class"""

    def __init__(self, configuration: dict[Any, Any]) -> None:
        self.configuration = configuration

    @property
    def catalog(self) -> CatalogInfo:
        """Set up attributes for STAC Catalog settings"""
        catalog_dict = self.configuration.get("catalog", {})
        catalog = CatalogInfo(
            name=catalog_dict.get("name", ""),
            url=catalog_dict.get("url", ""),
            rio_config=catalog_dict.get("rio_config", {}),
        )
        return catalog

    @property
    def collections(self) -> dict[Any, CollectionInfo]:
        """Set up attributes for STAC Collections settings"""
        collections_settings = self.configuration.get("collections", {})

        collections = {}
        for collection, settings in collections_settings.items():

            collections[collection] = CollectionInfo(
                id=collection,
                description=settings.get("description", ""),
                aliases=settings.get("aliases", {}),
                assets=settings.get("assets", {}),
                masks=settings.get("masks", {}),
            )

        return collections

    @property
    def masks(self) -> dict[str, dict[str, MaskInfo]]:
        """Set up dictionary of masks from STAC config"""

        # First find all products with masks
        collections_with_masks = [
            collection
            for collection, settings in self.collections.items()
            if settings.masks
        ]

        masks = {}

        # For each collection, loop over all masks and convert from dict to MaskInfo
        for collection in collections_with_masks:
            collection_masks_dicts = self.collections[collection].masks

            collection_masks = {}
            for mask, settings in collection_masks_dicts.items():

                alias = settings.get("alias", "")

                collection_masks[alias] = MaskInfo(
                    id=mask,
                    alias=alias,
                    description=settings.get("description", ""),
                    collection=collection,
                    type=settings.get("type", ""),
                    categories_to_mask=settings.get("categories_to_mask", []),
                    flags_definition=settings.get("flags_definition", {}),
                )
            masks[collection] = collection_masks

        return masks

    def __str__(self) -> str:
        return f"Configuration constructed from {self.configuration}"

    def __repr__(self) -> str:
        return f"STACConfig('{self.configuration}')"


def stac_config_from_toml(config_file_path) -> dict[Any, Any]:
    """Load the configuration dictionary from the TOML file"""
    with open(config_file_path, mode="rb") as f:
        config = tomli.load(f)

    return config
