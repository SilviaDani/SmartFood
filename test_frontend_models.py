#!/usr/bin/env python
"""Test che il frontend possa accedere ai modelli dal YAML"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/backend')

from smartfood.utils.model_registry import get_model_registry
import json

registry = get_model_registry()
models = registry.config_loader.get_all_models()

print("✅ Modelli disponibili nel YAML:\n")
print(json.dumps(models, indent=2))

print("\n\n✅ Formato per il frontend (API /api/models/detailed):\n")
frontend_models = [
    {
        "name": m["name"],
        "display_name": m["display_name"],
        "description": m.get("description", ""),
        "type": m.get("type", ""),
        "supports_confidence": m.get("supports_confidence", False)
    }
    for m in models
]
print(json.dumps({"success": True, "models": frontend_models, "count": len(frontend_models)}, indent=2))
