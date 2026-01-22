"""
Test del sistema di configurazione YAML per i modelli
"""

import sys
import os

# Aggiungi il percorso del backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from smartfood.utils.config_loader import get_config_loader
from smartfood.services.prediction_service import PredictionService


def test_config_loader():
    """Test del config loader"""
    print("\n" + "="*60)
    print("TEST: Config Loader")
    print("="*60)
    
    config_loader = get_config_loader()
    
    # Test 1: Carica i modelli
    print("\n✓ Test 1: Carica tutti i modelli")
    all_models = config_loader.get_all_models()
    for model in all_models:
        print(f"  - {model['name']}: {model['display_name']}")
    assert len(all_models) > 0, "Nessun modello caricato"
    
    # Test 2: Verifica supporto confidence
    print("\n✓ Test 2: Verifica supporto confidence")
    supports_chronos = config_loader.supports_confidence("chronos")
    print(f"  Chronos supports confidence: {supports_chronos}")
    assert supports_chronos == True, "Chronos dovrebbe supportare confidence"
    
    # Test 3: Ottieni threshold
    print("\n✓ Test 3: Ottieni threshold di confidenza")
    threshold = config_loader.get_confidence_threshold("chronos")
    print(f"  Chronos threshold: {threshold}")
    assert threshold == 0.45, "Threshold dovrebbe essere 0.45"
    
    # Test 4: Modelli con confidence
    print("\n✓ Test 4: Modelli con confidence")
    models_with = config_loader.get_models_with_confidence()
    print(f"  Modelli con confidence: {models_with}")
    assert "chronos" in models_with, "Chronos dovrebbe avere confidence"
    
    # Test 5: Configurazione visualizzazione
    print("\n✓ Test 5: Configurazione visualizzazione")
    vis_config = config_loader.get_visualization_config()
    print(f"  High threshold: {vis_config['high_threshold']}")
    print(f"  Medium threshold: {vis_config['medium_threshold']}")
    print(f"  Colori: {vis_config['confidence_color_scheme']}")
    
    print("\n✅ Tutti i test del config loader passati!")


def test_prediction_service():
    """Test del prediction service con formattazione confidenza"""
    print("\n" + "="*60)
    print("TEST: Prediction Service Formatting")
    print("="*60)
    
    # Crea il service
    uploads_folder = os.path.join(
        os.path.dirname(__file__),
        'backend/smartfood/uploads'
    )
    service = PredictionService(uploads_folder)
    
    # Test 1: Formatta predizione con confidence
    print("\n✓ Test 1: Formatta predizione con confidence")
    formatted = service.format_prediction(
        model_name="chronos",
        prediction_value=150,
        confidence=0.87
    )
    print(f"  Modello: {formatted['model_display_name']}")
    print(f"  Predizione: {formatted['prediction']} porzioni")
    print(f"  Confidenza: {formatted['confidence_percentage']}%")
    print(f"  Livello: {formatted['confidence_level']}")
    print(f"  Colore: {formatted['confidence_color']}")
    print(f"  Passa threshold: {formatted['passes_threshold']}")
    
    assert formatted['confidence'] == 0.87
    assert formatted['confidence_level'] == 'high'
    assert formatted['confidence_color'] == '#4CAF50'  # Verde
    
    # Test 2: Formatta predizione con bassa confidenza
    print("\n✓ Test 2: Formatta predizione con bassa confidenza")
    formatted_low = service.format_prediction(
        model_name="chronos",
        prediction_value=150,
        confidence=0.40
    )
    print(f"  Confidenza: {formatted_low['confidence_percentage']}%")
    print(f"  Livello: {formatted_low['confidence_level']}")
    print(f"  Colore: {formatted_low['confidence_color']}")
    print(f"  Passa threshold: {formatted_low['passes_threshold']}")
    
    assert formatted_low['confidence_level'] == 'low'
    assert formatted_low['confidence_color'] == '#F44336'  # Rosso
    assert formatted_low['passes_threshold'] == False
    
    # Test 3: Formatta predizione con confidenza media
    print("\n✓ Test 3: Formatta predizione con confidenza media")
    formatted_med = service.format_prediction(
        model_name="moment",
        prediction_value=120,
        confidence=0.65
    )
    print(f"  Confidenza: {formatted_med['confidence_percentage']}%")
    print(f"  Livello: {formatted_med['confidence_level']}")
    print(f"  Colore: {formatted_med['confidence_color']}")
    
    assert formatted_med['confidence_level'] == 'medium'
    assert formatted_med['confidence_color'] == '#FFC107'  # Giallo
    
    # Test 4: Filtra previsioni per confidenza
    print("\n✓ Test 4: Filtra previsioni per confidenza minima")
    predictions = [
        {"date": "2025-01-20", "portions": 120, "confidence": 0.90},
        {"date": "2025-01-21", "portions": 125, "confidence": 0.70},
        {"date": "2025-01-22", "portions": 130, "confidence": 0.40},
        {"date": "2025-01-23", "portions": 135, "confidence": 0.85},
    ]
    
    filtered = service.filter_predictions_by_confidence(
        predictions=predictions,
        min_confidence=0.75
    )
    
    print(f"  Totale predizioni: {len(predictions)}")
    print(f"  Dopo filtro (>= 0.75): {len(filtered)}")
    print(f"  Confidenze mantenute: {[p['confidence'] for p in filtered]}")
    
    assert len(filtered) == 2
    assert all(p['confidence'] >= 0.75 for p in filtered)
    
    print("\n✅ Tutti i test del prediction service passati!")


def test_confidence_levels():
    """Test della determinazione dei livelli di confidenza"""
    print("\n" + "="*60)
    print("TEST: Determinazione livelli di confidenza")
    print("="*60)
    
    service = PredictionService("")
    
    test_cases = [
        (0.95, "high"),
        (0.85, "high"),
        (0.80, "high"),
        (0.79, "medium"),
        (0.65, "medium"),
        (0.50, "medium"),
        (0.49, "low"),
        (0.30, "low"),
        (0.10, "low"),
    ]
    
    vis_config = {'high_threshold': 0.80, 'medium_threshold': 0.50}
    
    print("\nTestando classificazione di confidenza:")
    for confidence, expected_level in test_cases:
        level = service._get_confidence_level(confidence, vis_config)
        status = "✓" if level == expected_level else "✗"
        print(f"  {status} {confidence:.2f} → {level} (atteso: {expected_level})")
        assert level == expected_level
    
    print("\n✅ Tutti i test dei livelli di confidenza passati!")


if __name__ == "__main__":
    try:
        test_config_loader()
        test_prediction_service()
        test_confidence_levels()
        
        print("\n" + "="*60)
        print("🎉 TUTTI I TEST PASSATI!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
