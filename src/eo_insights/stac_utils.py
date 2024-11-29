"""Construct a STAC config object"""

import pathlib
from dataclasses import dataclass
from typing import Any, Union

import numpy
import tomllib

from eo_insights.utils import get_logger

PathType = Union[str, pathlib.Path]


@dataclass
class CatalogInfo:
    """Data class for a STAC Catalog"""

    name: str
    url: str
    rio_config: dict


@dataclass
class MaskInfo:
    """Data class for masking information from STAC config"""

    id: str
    alias: str
    description: str
    collection: str
    mask_type: str
    default_masking_settings: dict[str, Any]
    flags_definition: dict[str, dict[str, Any]]


def _get_mask_info(collection_id: str, mask_id: str, mask_config: dict) -> MaskInfo:
    """Converts from configuration dictionary to a MaskInfo item"""

    mask = MaskInfo(
        id=mask_id,
        alias=mask_config.get("alias", ""),
        description=mask_config.get("description", ""),
        collection=collection_id,
        mask_type=mask_config.get("type", ""),
        default_masking_settings=mask_config.get("default_masking_settings", {}),
        flags_definition=mask_config.get("flags_definition", {}),
    )

    return mask


@dataclass
class CollectionInfo:
    """Data class for a STAC Collection"""

    id: str
    description: str
    aliases: dict
    assets: dict
    masks: dict[str, MaskInfo]


def _get_collection_info(collection_id: str, collection_config: dict) -> CollectionInfo:
    """Converts from configuration dictionary to a CollectionInfo item"""

    # Process mask information for the collection
    masks_config = collection_config.get("masks", {})

    masks = {
        mask_info.get("alias", mask_id): _get_mask_info(
            collection_id, mask_id, mask_info
        )
        for mask_id, mask_info in masks_config.items()
    }

    collection = CollectionInfo(
        id=collection_id,
        description=collection_config.get("description", ""),
        aliases=collection_config.get("aliases", {}),
        assets=collection_config.get("assets", {}),
        masks=masks,
    )

    return collection


class STACConfig:
    """STAC config class"""

    _log = None

    def __init__(self, configuration: dict[Any, Any]) -> None:
        self.configuration = configuration
        self._log = get_logger("stac config")

    @classmethod
    def from_toml(cls, configuration_toml_path: PathType):
        """Load the configuration from a TOML file"""
        with open(configuration_toml_path, mode="rb") as f:
            config_dict = tomllib.load(f)

        # Specifically need to replace NaN with np.nan
        for collection, settings in config_dict.get("collections", {}).items():
            for asset, asset_settings in settings.get("assets", {}).items():
                nodata = asset_settings.get("nodata", "")
                if isinstance(nodata, str):
                    if nodata.lower() == "nan":
                        config_dict["collections"][collection]["assets"][asset][
                            "nodata"
                        ] = numpy.nan
                    else:
                        raise NotImplementedError(
                            "Only NaN, nan, and NAN are accepted as string values for an asset's nodata value."
                            "Check the configuration file."
                        )

        return cls(config_dict)

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
    def collections(self) -> dict[str, CollectionInfo]:
        """Set up attributes for STAC Collections settings"""
        collections_config = self.configuration.get("collections", {})

        collections = {
            collection_id: _get_collection_info(collection_id, collection_config)
            for collection_id, collection_config in collections_config.items()
        }

        return collections

    def list_collections(self):
        """Display the collection information"""
        for collection_name, collection_info in self.collections.items():
            self._log.info("%s - %s", collection_name, collection_info.description)

    def get_collection_masks(self, collection_id: str) -> dict[str, MaskInfo]:
        """Get a dictionary of alias: MaskInfo for all masks available in a collection"""
        collection_info = self.collections.get(collection_id, None)

        if collection_info is not None:
            masks = collection_info.masks

            mask_aliases = {
                mask_info.alias: mask_info for _, mask_info in masks.items()
            }
        else:
            mask_aliases = {}
            self._log.error("No collection information was found for %s", collection_id)

        return mask_aliases

    def __str__(self) -> str:
        return f"Configuration constructed from {self.configuration}"

    def __repr__(self) -> str:
        return f"STACConfig('{self.configuration}')"
