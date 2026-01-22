import React, { useEffect, useState } from 'react';
import './PredictionDisplay.css';

interface PredictionData {
  model: string;
  model_display_name: string;
  prediction: number;
  supports_confidence: boolean;
  confidence?: number;
  confidence_percentage?: number;
  confidence_level?: 'high' | 'medium' | 'low';
  confidence_color?: string;
  passes_threshold?: boolean;
}

interface Props {
  data: PredictionData;
  showConfidence?: boolean;
}

/**
 * Componente per visualizzare una predizione singola
 * 
 * Mostra:
 * - Nome del modello
 * - Valore della predizione
 * - Confidence score (se il modello lo supporta)
 * - Colore e livello di confidenza
 */
export const PredictionDisplay: React.FC<Props> = ({ 
  data, 
  showConfidence = true 
}) => {
  const [modelConfig, setModelConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Carica la configurazione del modello
    fetch('/api/config/models')
      .then(r => r.json())
      .then(response => {
        if (response.success) {
          const modelCfg = response.models.find(
            (m: any) => m.name === data.model
          );
          setModelConfig(modelCfg);
        }
      })
      .catch(err => console.error('Error loading model config:', err))
      .finally(() => setLoading(false));
  }, [data.model]);

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3 className="model-name">
          {data.model_display_name || data.model}
        </h3>
      </div>

      <div className="prediction-value-container">
        <div className="prediction-value">
          {data.prediction.toFixed(0)}
        </div>
        <div className="prediction-unit">portions</div>
      </div>

      {showConfidence && data.supports_confidence && data.confidence !== undefined && (
        <div className="confidence-container">
          <div className="confidence-header">
            <span className="confidence-label">Confidence</span>
            <span className="confidence-percentage">
              {data.confidence_percentage}%
            </span>
          </div>

          <div className="confidence-bar-wrapper">
            <div
              className="confidence-bar"
              style={{
                width: `${data.confidence_percentage}%`,
                backgroundColor: data.confidence_color || '#4CAF50',
                transition: 'width 0.3s ease',
              }}
            />
          </div>

          <div className={`confidence-badge ${data.confidence_level || 'medium'}`}>
            {(data.confidence_level || 'UNKNOWN').toUpperCase()}
          </div>

          {!data.passes_threshold && (
            <div className="threshold-warning">
              ⚠️ Below confidence threshold
            </div>
          )}
        </div>
      )}

      {showConfidence && !data.supports_confidence && (
        <div className="no-confidence-notice">
          Model does not provide confidence scores
        </div>
      )}
    </div>
  );
};

export default PredictionDisplay;
