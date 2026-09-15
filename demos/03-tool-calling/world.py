"""Small in-memory world and the six tools exposed to the model."""
import copy
import math

from jsonschema import ValidationError, validate

TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
POSITION = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
            "description": "Absolute [x, y, z] coordinates; y is up."}


def tool(name, summary, **properties):
    return {"type": "function", "function": {"name": name, "description": summary,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


TOOLS = [
    tool("search_asset", "Search the local asset catalog by a short Korean or English keyword. Empty matches are possible.", query=TEXT),
    tool("generate_asset", "Create placeholder asset metadata from a description. Does not create a real 3D model or place it.", description=TEXT),
    tool("place_asset", "Place a new instance of an existing asset. Returns a new object ID.", asset_id=TEXT, position=POSITION),
    tool("move_asset", "Move a placed object to an absolute position using its object ID.", object_id=TEXT, position=POSITION),
    tool("delete_asset", "Remove a placed object using its object ID; keep its catalog asset.", object_id=TEXT),
    tool("inspect_scene", "Return current assets and placed objects, including IDs and coordinates."),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


class World:
    def __init__(self):
        self.assets = {
            "tree": {"id": "tree", "description": "나무 tree oak", "source": "catalog"},
            "rock": {"id": "rock", "description": "바위 돌 rock stone", "source": "catalog"},
            "house": {"id": "house", "description": "집 오두막 house cabin", "source": "catalog"},
        }
        self.objects = {}
        self._next_asset = 1
        self._next_object = 1

    def execute(self, name, arguments):
        """Validate model input before mutation; return JSON-compatible results."""
        if name not in SCHEMAS:
            return {"error": f"Unknown tool: {name}"}
        try:
            validate(arguments, SCHEMAS[name])
            position = arguments.get("position")
            if position is not None and not all(math.isfinite(v) for v in position):
                raise ValueError("Coordinates must be finite numbers")
            result = getattr(self, name)(**arguments)
            return copy.deepcopy(result)
        except ValidationError as exc:
            return {"error": exc.message}
        except (ValueError, KeyError, OverflowError) as exc:
            return {"error": str(exc)}

    def search_asset(self, query):
        tokens = query.casefold().split()
        return {"assets": [a for a in self.assets.values()
                           if any(t in (a["id"] + " " + a["description"]).casefold() for t in tokens)]}

    def generate_asset(self, description):
        asset_id = f"generated_{self._next_asset}"
        self._next_asset += 1
        asset = {"id": asset_id, "description": description, "source": "generated_placeholder"}
        self.assets[asset_id] = asset
        return asset

    def place_asset(self, asset_id, position):
        if asset_id not in self.assets:
            raise ValueError(f"Unknown asset_id: {asset_id}. Search or generate an asset first.")
        object_id = f"object_{self._next_object}"
        self._next_object += 1
        obj = {"id": object_id, "asset_id": asset_id, "position": list(position)}
        self.objects[object_id] = obj
        return obj

    def move_asset(self, object_id, position):
        if object_id not in self.objects:
            raise ValueError(f"Unknown object_id: {object_id}. Inspect the scene first.")
        self.objects[object_id]["position"] = list(position)
        return self.objects[object_id]

    def delete_asset(self, object_id):
        if object_id not in self.objects:
            raise ValueError(f"Unknown object_id: {object_id}. Inspect the scene first.")
        return {"deleted": self.objects.pop(object_id)}

    def inspect_scene(self):
        return {"assets": self.assets, "objects": self.objects}
