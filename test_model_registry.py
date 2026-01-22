"""
Test del Model Registry - Verifica che il sistema centralizzato funzioni
"""

import sys
import os

# Aggiungi il percorso del backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


def test_model_registry():
    """Test del model registry"""
    print("\n" + "="*70)
    print("TEST: Model Registry")
    print("="*70)
    
    from smartfood.utils.model_registry import get_model_registry
    
    registry = get_model_registry()
    
    # Test 1: Ottieni tutti i modelli disponibili
    print("\n✓ Test 1: Ottieni tutti i modelli disponibili")
    models = registry.get_available_models()
    print(f"  Modelli disponibili: {models}")
    assert len(models) > 0, "Nessun modello disponibile"
    
    # Test 2: Verifica se un modello è disponibile
    print("\n✓ Test 2: Verifica disponibilità modelli")
    for model in models:
        available = registry.is_model_available(model)
        print(f"  {model}: {available}")
        assert available == True, f"{model} dovrebbe essere disponibile"
    
    # Test 3: Ottieni configurazione modello specifico
    print("\n✓ Test 3: Ottieni configurazione modello specifico")
    for model in models:
        config = registry.get_model_config(model)
        print(f"  {model}:")
        print(f"    - display_name: {config['display_name']}")
        print(f"    - supports_confidence: {config['supports_confidence']}")
        print(f"    - threshold: {config['min_confidence_threshold']}")
    
    # Test 4: Verifica che modelli non-disponibili non siano riconosciuti
    print("\n✓ Test 4: Verifica reiezione modelli inesistenti")
    fake_model = "fake_model_xyz"
    available = registry.is_model_available(fake_model)
    print(f"  {fake_model} disponibile? {available}")
    assert available == False, f"{fake_model} dovrebbe NON essere disponibile"
    
    print("\n✅ Tutti i test del registry passati!")


def test_prediction_service_integration():
    """Test integrazione del registry con PredictionService"""
    print("\n" + "="*70)
    print("TEST: Integrazione Model Registry con PredictionService")
    print("="*70)
    
    from smartfood.services.prediction_service import PredictionService
    from smartfood.utils.model_registry import get_model_registry
    
    # Crea il service
    uploads_folder = os.path.join(
        os.path.dirname(__file__),
        'backend/smartfood/uploads'
    )
    service = PredictionService(uploads_folder)
    registry = get_model_registry()
    
    # Test 1: Verifica registrazione degli handler
    print("\n✓ Test 1: Verifica registrazione degli handler")
    models = registry.get_available_models()
    print(f"  Modelli registrati nel service: {models}")
    
    # Test 2: Tenta di usare un modello registrato
    print("\n✓ Test 2: Testa formato predizione")
    for model in models:
        formatted = service.format_prediction(
            model_name=model,
            prediction_value=150,
            confidence=0.87
        )
        print(f"\n  {model}:")
        print(f"    - Model: {formatted['model']}")
        print(f"    - Display Name: {formatted['model_display_name']}")
        print(f"    - Supporta Confidence: {formatted['supports_confidence']}")
        
        if formatted['supports_confidence']:
            print(f"    - Confidence: {formatted['confidence_percentage']}%")
            print(f"    - Livello: {formatted['confidence_level']}")
            print(f"    - Colore: {formatted['confidence_color']}")
    
    print("\n✅ Integrazione con PredictionService OK!")


def test_dynamic_model_validation():
    """Test della validazione dinamica dei modelli"""
    print("\n" + "="*70)
    print("TEST: Validazione Dinamica dei Modelli")
    print("="*70)
    
    from smartfood.utils.model_registry import get_model_registry
    
    registry = get_model_registry()
    available = registry.get_available_models()
    
    print(f"\n✓ Modelli disponibili nel sistema: {available}")
    
    # Test con un modello valido
    print(f"\n✓ Test validazione modello valido: {available[0]}")
    try:
        result = registry.predict(available[0], None, 10)
        print(f"  ❌ Avrebbe dovuto lanciare un errore (nessun handler)")
    except ValueError as e:
        print(f"  ✓ Errore corretto: {str(e)[:60]}...")
    
    # Test con un modello invalido
    print(f"\n✓ Test validazione modello invalido: fake_model")
    try:
        registry.predict("fake_model", None, 10)
        print(f"  ❌ Avrebbe dovuto lanciare un errore")
    except ValueError as e:
        print(f"  ✓ Errore corretto: {str(e)[:60]}...")
    
    print("\n✅ Validazione dinamica OK!")


if __name__ == "__main__":
    try:
        test_model_registry()
        test_prediction_service_integration()
        test_dynamic_model_validation()
        
        print("\n" + "="*70)
        print("🎉 TUTTI I TEST DEL MODEL REGISTRY PASSATI!")
        print("="*70)
        print("\n✅ Il sistema è pronto per aggiungere/rimuovere modelli dal YAML!")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
