"""
Model Registry - Gestore centralizzato dei modelli disponibili
Legge la configurazione YAML e fornisce accesso ai modelli
"""

from smartfood.utils.config_loader import get_config_loader
from typing import List, Optional, Callable, Dict, Any


class ModelRegistry:
    """
    Registro centralizzato dei modelli.
    Legge tutti i modelli dal YAML e fornisce metodi per accedervi.
    """
    
    def __init__(self):
        """Inizializza il registro leggendo la configurazione"""
        self.config_loader = get_config_loader()
        self._prediction_handlers: Dict[str, Callable] = {}
        self._training_handlers: Dict[str, Callable] = {}
    
    def register_prediction_handler(self, model_name: str, handler: Callable):
        """
        Registra un handler per la predizione di un modello
        
        Args:
            model_name: Nome del modello (da YAML)
            handler: Funzione che esegue la predizione
        """
        self._prediction_handlers[model_name.lower()] = handler
    
    def register_training_handler(self, model_name: str, handler: Callable):
        """
        Registra un handler per l'addestramento di un modello
        
        Args:
            model_name: Nome del modello (da YAML)
            handler: Funzione che esegue l'addestramento
        """
        self._training_handlers[model_name.lower()] = handler
    
    def get_available_models(self) -> List[str]:
        """
        Ritorna la lista di tutti i modelli disponibili nel YAML
        
        Returns:
            Lista dei nomi dei modelli (es: ['chronos', 'moment', 'timesfm'])
        """
        return [model['name'] for model in self.config_loader.get_all_models()]
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Verifica se un modello è disponibile nel YAML
        
        Args:
            model_name: Nome del modello
        
        Returns:
            True se il modello è nel YAML, False altrimenti
        """
        return model_name.lower() in [m['name'].lower() for m in self.config_loader.get_all_models()]
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Ritorna la configurazione completa di un modello
        
        Args:
            model_name: Nome del modello
        
        Returns:
            Dict con la configurazione o None se non trovato
        """
        return self.config_loader.get_model_config(model_name)
    
    def predict(self, model_name: str, *args, **kwargs) -> any:
        """
        Esegue una predizione usando l'handler registrato
        
        Args:
            model_name: Nome del modello
            *args, **kwargs: Parametri per il handler
        
        Returns:
            Risultato della predizione
        
        Raises:
            ValueError: Se il modello non è disponibile o nessun handler registrato
        """
        model_lower = model_name.lower()
        
        if not self.is_model_available(model_name):
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Available models: {', '.join(self.get_available_models())}"
            )
        
        if model_lower not in self._prediction_handlers:
            raise ValueError(
                f"No prediction handler registered for model '{model_name}'. "
                f"Make sure the model is registered using register_prediction_handler()"
            )
        
        return self._prediction_handlers[model_lower](*args, **kwargs)
    
    def train(self, model_name: str, *args, **kwargs) -> any:
        """
        Esegue l'addestramento usando l'handler registrato
        
        Args:
            model_name: Nome del modello
            *args, **kwargs: Parametri per il handler
        
        Returns:
            Risultato dell'addestramento
        
        Raises:
            ValueError: Se il modello non è disponibile o nessun handler registrato
        """
        model_lower = model_name.lower()
        
        if not self.is_model_available(model_name):
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Available models: {', '.join(self.get_available_models())}"
            )
        
        if model_lower not in self._training_handlers:
            raise ValueError(
                f"No training handler registered for model '{model_name}'. "
                f"Make sure the model is registered using register_training_handler()"
            )
        
        return self._training_handlers[model_lower](*args, **kwargs)


# Istanza singleton globale
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Ritorna l'istanza singleton del ModelRegistry
    
    Returns:
        Istanza globale del registry
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
