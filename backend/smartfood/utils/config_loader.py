"""
Config Loader - Carica la configurazione YAML dei modelli
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class ModelConfigLoader:
    def __init__(self, config_path: str = None):
        """
        Inizializza il loader con il percorso della configurazione
        
        Args:
            config_path: Percorso al file YAML (default: backend/config/models_confidence.yml)
        """
        if config_path is None:
            # Default path relativo alla cartella backend
            config_path = os.path.join(
                Path(__file__).parent.parent.parent,
                'config',
                'models_confidence.yml'
            )
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.models: Dict[str, Dict] = {}
        self.visualization: Dict[str, Any] = {}
        
        self._load_config()
    
    def _load_config(self):
        """Carica e parsa il file YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Popola il dizionario dei modelli per accesso veloce
        for model in self.config.get('models', []):
            self.models[model['name']] = model
        
        self.visualization = self.config.get('visualization', {})
    
    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """Ritorna la configurazione di un modello specifico"""
        return self.models.get(model_name.lower())
    
    def supports_confidence(self, model_name: str) -> bool:
        """Verifica se un modello supporta confidence scores"""
        config = self.get_model_config(model_name)
        return config.get('supports_confidence', False) if config else False
    
    def get_confidence_threshold(self, model_name: str) -> float:
        """Ritorna il threshold minimo di confidenza per un modello"""
        config = self.get_model_config(model_name)
        return config.get('min_confidence_threshold', 0.5) if config else 0.5
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Ritorna la configurazione di visualizzazione"""
        return self.visualization
    
    def get_all_models(self) -> List[Dict]:
        """Ritorna la lista di tutti i modelli"""
        return self.config.get('models', [])
    
    def get_models_with_confidence(self) -> List[str]:
        """Ritorna la lista dei modelli che supportano confidence"""
        return [
            model['name'] for model in self.config.get('models', [])
            if model.get('supports_confidence', False)
        ]
    
    def get_models_without_confidence(self) -> List[str]:
        """Ritorna la lista dei modelli senza confidence"""
        return [
            model['name'] for model in self.config.get('models', [])
            if not model.get('supports_confidence', False)
        ]
    
    def reload(self):
        """Ricarica la configurazione dal file YAML"""
        self._load_config()


# Istanza globale del loader (singleton pattern)
_config_loader: Optional[ModelConfigLoader] = None


def get_config_loader() -> ModelConfigLoader:
    """Ritorna l'istanza singleton del config loader"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ModelConfigLoader()
    return _config_loader
