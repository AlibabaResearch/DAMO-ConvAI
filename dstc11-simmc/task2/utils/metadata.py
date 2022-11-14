import json
import os
import re
from typing import List

import attr
from attr.validators import instance_of

DATA_DIR = ''  # give root folder name of simmc2 as argument. Ex) find_data_dir('DSTC10-SIMMC')

'''
    Fashion Metadata
'''

ERROR_SCENE_LIST = [
    "cloth_store_1416238_woman_20_6",
    "cloth_store_1416238_woman_4_8",
    "m_cloth_store_1416238_woman_20_6",
    "cloth_store_1416238_woman_19_0"
]

FASHION_SIZES = [
    "XS",
    "S",
    "M",
    "L",
    "XL",
    "XXL"
]

FASHION_AVAILABLE_SIZES = [
    "<A>",
    "<B>",
    "<C>",
    "<D>",
    "<E>",
    "<F>"
]

FASHION_BRAND = [
    "212 Local",
    "Art Den",
    "Art News Today",
    "Brain Puzzles",
    "Cats Are Great",
    "Coats & More",
    "Downtown Consignment",
    "Downtown Stylists",
    "Fancy Nails",
    "Garden Retail",
    "Glam Nails",
    "Global Voyager",
    "HairDo",
    "Home Store",
    "Modern Arts",
    "Nature Photographers",
    "New Fashion",
    "North Lodge",
    "Ocean Wears",
    "Pedals & Gears",
    "River Chateau",
    "StyleNow Feed",
    "The Vegan Baker",
    "Uptown Gallery",
    "Uptown Studio",
    "Yogi Fit"
]

FASHION_COLOR = [
    "beige",
    "black",
    "black, grey",
    "black, olive",
    "black, orange",
    "black, red",
    "black, red, white",
    "black, white",
    "blue",
    "blue, black",
    "blue, green",
    "blue, grey",
    "blue, white",
    "brown",
    "brown, black",
    "brown, white",
    "dark blue",
    "dark brown",
    "dark green",
    "dark green, dark blue",
    "dark grey",
    "dark pink",
    "dark pink, white",
    "dark red",
    "dark violet",
    "dark yellow",
    "dirty green",
    "dirty grey",
    "golden",
    "green",
    "green, black",
    "green, violet, pink",
    "green, white",
    "grey",
    "grey, black",
    "grey, blue",
    "grey, brown",
    "grey, white",
    "light blue",
    "light blue, light green",
    "light grey",
    "light orange",
    "light pink",
    "light red",
    "maroon",
    "maroon, white, blue",
    "olive",
    "olive, black",
    "olive, white",
    "orange",
    "orange, purple",
    "pink",
    "pink, black",
    "pink, white",
    "purple",
    "red",
    "red, black",
    "red, grey",
    "red, white",
    "red, white, yellow",
    "violet",
    "white",
    "white, black",
    "white, black, red",
    "white, blue",
    "white, grey",
    "white, red, violet",
    "yellow",
    "yellow, black",
    "yellow, brown",
    "yellow, white"
]

FASHION_PATTERN = [
    "camouflage",
    "canvas",
    "cargo",
    "checkered",
    "checkered, plain",
    "denim",
    "design",
    "diamonds",
    "dotted",
    "floral",
    "heavy stripes",
    "heavy vertical stripes",
    "holiday",
    "horizontal stripes",
    "knit",
    "leafy design",
    "leapard print",
    "leather",
    "light spots",
    "light stripes",
    "light vertical stripes",
    "multicolored",
    "plaid",
    "plain",
    "plain with stripes on side",
    "radiant",
    "spots",
    "star design",
    "streaks",
    "stripes",
    "text",
    "twin colors",
    "velvet",
    "vertical design",
    "vertical stripes",
    "vertical striples"
]

FASHION_SLEEVE_LENGTH = [
    "",
    "full",
    "half",
    "long",
    "short",
    "sleeveless"
]

FASHION_ASSET_TYPE = [
    "blouse_display",
    "blouse_hanging",
    "dress_hanging",
    "jacket_display",
    "jacket_hanging",
    "tshirt_display",
    "tshirt_folded",
    "trousers_display",
    "tshirt_hanging",
    "hat",
    "shoes",
    "skirt"
]

FASHION_TYPE = [
    "blouse",
    "coat",
    "dress",
    "hat",
    "hoodie",
    "jacket",
    "jeans",
    "joggers",
    "shirt",
    "shirt, vest",
    "shoes",
    "skirt",
    "suit",
    "sweater",
    "tank top",
    "trousers",
    "tshirt",
    "vest"
]

# delex
FASHION_PRICE = ['4.99', '9.99', '14.99', '19.99', '24.99', '29.99', '34.99', '39.99', '44.99', '49.99', '54.99', '59.99', '64.99', '69.99', '74.99', '79.99', '84.99', '89.99', '94.99', '99.99', '109.99', '114.99', '119.99', '124.99', '129.99', '134.99', '139.99', '144.99', '149.99', '154.99', '164.99', '169.99', '174.99', '179.99', '184.99', '189.99', '199.99', '204.99', '209.99', '214.99', '224.99', '229.99', '234.99', '239.99', '244.99']

# delex
FASHION_CUSTOMER_REVIEW = ['2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9', '5.0']

'''
    Furniture Metadata
'''

FURNITURE_BRAND = [
    "212 Local",
    "Art Den",
    "Downtown Consignment",
    "Downtown Stylists",
    "Global Voyager",
    "Home Store",
    "Modern Arts",
    "North Lodge",
    "River Chateau",
    "StyleNow Feed",
    "Uptown Gallery",
    "Uptown Studio"
]

FURNITURE_COLOR = [
    "black",
    "black and white",
    "blue",
    "brown",
    "green",
    "grey",
    "red",
    "white",
    "wooden"
]

FURNITURE_MATERIALS = [
    "leather",
    "marble",
    "memory foam",
    "metal",
    "natural fibers",   
    "wood",
    "wool"
]

FURNITURE_TYPE = [
    "AreaRug",
    "Bed",
    "Chair",
    "CoffeeTable",
    "CouchChair",
    "EndTable",
    "Lamp",
    "Shelves",
    "Sofa",
    "Table"
]

# delex
FURNITURE_PRICE = ['$199', '$249', '$299', '$349', '$399', '$449', '$499', '$549', '$599', '$649']

# delex
FURNITURE_CUSTOMER_RATING = ['2.7', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '4.2', '4.3', '4.4', '4.7', '4.8', '4.9', '5.0']

@attr.s 
class FashionMetadata:
    name: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    asset_type: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    customer_review: float = attr.ib(
        converter=float,
        validator=instance_of(float)
    )
    available_sizes: List[str] = attr.ib(
        converter=lambda x: [str(_) for _ in x],
        validator=instance_of(list)
    )
    color: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    pattern: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    brand: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    sleeve_length: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    type: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    price: float = attr.ib(
        converter=float,
        validator=instance_of(float)
    )
    size: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )

    @staticmethod
    def check_in(attribute, value, listing):
        """Universal checker that validates if value is in the given list."""
        if value not in listing:
            raise ValueError("{} must be one of {}, but received {}.".format(attribute.name, listing, value))
    
    @asset_type.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FASHION_ASSET_TYPE)
    
    @brand.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FASHION_BRAND)

    @pattern.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FASHION_PATTERN)
    
    @color.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FASHION_COLOR)
    
    @sleeve_length.validator
    def check(self, attribute, value):
        self.check_in(attribute ,value, FASHION_SLEEVE_LENGTH)

    @type.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FASHION_TYPE)

    @available_sizes.validator
    def check(self, attribute, value):
        common = set(value) & set(FASHION_SIZES)
        if len(common) < 1:
            raise ValueError("Available sizes must be one of {}, but receieved {}.".format(FASHION_SIZES, value))

    @size.validator
    def check(self, attribute, value):
        listing = getattr(self, "available_sizes")
        self.check_in(attribute, value, listing)

    @customer_review.validator
    def check(self, attribute, value):
        if not (0.0 <= value <= 5.0):
            raise ValueError("Rating must be in range [0.0, 5.0].")
@attr.s
class FurnitureMetadata:
    name: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    brand: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    color: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    customer_review: float = attr.ib(
        converter=float,
        validator=instance_of(float)
    )
    materials: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    price: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )
    type: str = attr.ib(
        converter=str,
        validator=instance_of(str)
    )

    @staticmethod
    def check_in(attribute, value, listing):
        """Universal checker that validates if value is in the given list."""
        if value not in listing:
            raise ValueError("{} must be one of {}, but received {}.".format(attribute.name, listing, value))

    @brand.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FURNITURE_BRAND)
    
    @color.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FURNITURE_COLOR)

    @materials.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FURNITURE_MATERIALS)

    @type.validator
    def check(self, attribute, value):
        self.check_in(attribute, value, FURNITURE_TYPE)

    @customer_review.validator
    def check(self, attribute, value):
        if not (0.0 <= value <= 5.0):
            raise ValueError("Rating must be in range [0.0, 5.0].")


def main_function():
    """
        Converts each key from CamelCase to snake_case.
        Also changes some key names to be more consistent
        across dataset.
    """
    _underscore1 = re.compile(r'(.)([A-Z][a-z]+)')
    _underscore2 = re.compile(r'([a-z0-9])([A-Z])')
    def key_map(key):
        subbed = _underscore1.sub(r'\1_\2', key)
        subbed = _underscore2.sub(r'\1_\2', subbed).lower()
        if subbed in ("customer_review", "customer_rating"):
            return "customer_review"
        return subbed

    FASHION_JSON = os.path.join(DATA_DIR, "fashion_prefab_metadata_all.json")
    FURNITURE_JSON = os.path.join(DATA_DIR, "furniture_prefab_metadata_all.json")

    fashion_items = json.load(open(FASHION_JSON, 'r'))
    fashion_items = [
        FashionMetadata(
            name=key,
            **{key_map(k): v for k,v in value.items()}
        ) for key, value in fashion_items.items()
    ]

    furniture_items = json.load(open(FURNITURE_JSON, 'r'))
    furniture_items = [
        FurnitureMetadata(
            name=key,
            **{key_map(k): v for k,v in value.items()}
        ) for key, value in furniture_items.items()
    ]
    
    return fashion_items, furniture_items 


def load_metadata(data_dir):
    """
        Converts each key from CamelCase to snake_case.
        Also changes some key names to be more consistent
        across dataset.
    """
    _underscore1 = re.compile(r'(.)([A-Z][a-z]+)')
    _underscore2 = re.compile(r'([a-z0-9])([A-Z])')
    def key_map(key):
        subbed = _underscore1.sub(r'\1_\2', key)
        subbed = _underscore2.sub(r'\1_\2', subbed).lower()
        if subbed in ("customer_review", "customer_rating"):
            return "customer_review"
        return subbed

    FASHION_JSON = os.path.join(data_dir, "fashion_prefab_metadata_all.json")
    FURNITURE_JSON = os.path.join(data_dir, "furniture_prefab_metadata_all.json")

    fashion_items = json.load(open(FASHION_JSON, 'r'))
    fashion_items = [
        FashionMetadata(
            name=key,
            **{key_map(k): v for k,v in value.items()}
        ) for key, value in fashion_items.items()
    ]

    furniture_items = json.load(open(FURNITURE_JSON, 'r'))
    furniture_items = [
        FurnitureMetadata(
            name=key,
            **{key_map(k): v for k,v in value.items()}
        ) for key, value in furniture_items.items()
    ]
    
    return fashion_items, furniture_items 

fashion_meta_attrs = {
    'size': FASHION_SIZES,
    'available_sizes': FASHION_AVAILABLE_SIZES,
    'brand': FASHION_BRAND,
    'color': FASHION_COLOR,
    'pattern': FASHION_PATTERN,
    'sleeve_length': FASHION_SLEEVE_LENGTH,
    'asset_type': FASHION_ASSET_TYPE,
    'type': FASHION_TYPE,
    'price': FASHION_PRICE,
    'customer_review': FASHION_CUSTOMER_REVIEW,
}

furniture_meta_attrs = {
    'brand': FURNITURE_BRAND,
    'color': FURNITURE_COLOR,
    'materials': FURNITURE_MATERIALS,
    'type': FURNITURE_TYPE,
    'price': FURNITURE_PRICE,
    'customer_review': FURNITURE_CUSTOMER_RATING  # key is "review"!!
}

available_sizes2st = {
    'XS': '<A>',
    'S': '<B>',
    'M': '<C>',
    'L': '<D>',
    'XL': '<E>',
    'XXL': '<F>'
}

if __name__ == "__main__":

    from pprint import pprint
    fashion_items, furniture_items = main_function()

    print("Fashion items (first 5) : ")
    pprint(fashion_items[:5])

    print("Furniture items (first 5) : ")
    pprint(furniture_items[:5])

    # import pickle

    # with open("fashion_meta.pkl", 'wb') as f:
    #     pickle.dump(fashion_items, f)
    # with open("furniture_meta.pkl", 'wb') as f:
    #     pickle.dump(furniture_items, f)
   
