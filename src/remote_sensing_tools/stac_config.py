"""Construct a STAC config object"""

import tomli


class STACConfig:
    """STAC config class"""

    def __init__(self, config_file_path=None) -> None:
        if config_file_path is not None:
            self.config_dictionary = self.config_dictionary_from_toml(config_file_path)
            self.stac_settings = self.config_dictionary.get("stac", {})
            self.catalog_url = self.stac_settings.get("catalog_url")
            self.rio_config = self.stac_settings.get("rio_config")
            self.products = self.stac_settings.get("configured_products", {})
        else:
            self.config_dictionary = None

    def config_dictionary_from_toml(self, config_file_path):
        """Construct the configuration dictionary"""
        with open(config_file_path, mode="rb") as f:
            config = tomli.load(f)

        return config

    def get_product_dictionary(self, product):
        """Extract configuration information for a single product"""

        if product is not None:
            product_dict = self.config_dictionary.get(product, {})
        else:
            product_dict = None

        return product_dict

    def get_pq_mask_dictionary(self, product):
        """Extract masking information for a single product"""

        product_dict = self.get_product_dictionary(product)

        if product_dict is not None:
            masking_dict = product_dict.get("pq_mask", {})
        else:
            masking_dict = None

        return masking_dict
