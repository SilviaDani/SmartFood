import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Brain, Play, Square, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

interface AITrainingProps {
  onLogout?: () => void;
}

const MODELS = [
  { id: 'moment', name: 'MOMENT', description: 'Multi-horizon time series forecasting' },
  { id: 'chronos', name: 'Chronos', description: 'Probabilistic time series forecasting' },
];

export function AITraining({ onLogout }: AITrainingProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [csvFiles, setCsvFiles] = useState<string[]>([]);
  const [trainingJob, setTrainingJob] = useState<any>(null);

  // Carica la lista dei file CSV disponibili
  useEffect(() => {
    loadAvailableDatasets();
  }, []);

  const loadAvailableDatasets = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/datasets`);
      
      if (response.ok) {
        const data = await response.json();
        setCsvFiles(data.files || []);
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
      // Fallback: mostra un messaggio
      toast.error('Could not load available datasets');
    }
  };

  const startTraining = async () => {
    if (!selectedModel) {
      toast.error('Please select a model');
      return;
    }
    
    if (!selectedDataset) {
      toast.error('Please select a dataset');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      
      const response = await fetch(`${apiUrl}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          dataset_id: selectedDataset,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to start training');
      }

      const data = await response.json();
      setTrainingJob(data);
      
      toast.success('Training started! Monitoring progress...');
      
      // Poll for progress
      pollTrainingProgress(data.job_id);
    } catch (error) {
      console.error('Training error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to start training');
      setIsTraining(false);
    }
  };

  const pollTrainingProgress = (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/train/${jobId}/status`);
        
        if (!response.ok) {
          throw new Error('Failed to get training status');
        }

        const data = await response.json();
        setTrainingProgress(data.progress || 0);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setIsTraining(false);
          setTrainingJob(data);
          toast.success(`Training completed! Accuracy: ${data.results?.accuracy?.toFixed(2)}%`);
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setIsTraining(false);
          toast.error(`Training failed: ${data.error}`);
        }
      } catch (error) {
        console.error('Poll error:', error);
        clearInterval(pollInterval);
        setIsTraining(false);
      }
    }, 2000); // Poll every 2 seconds
  };

  const stopTraining = () => {
    setIsTraining(false);
    setTrainingProgress(0);
    setTrainingJob(null);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI Model Training
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label htmlFor="model-select">Select Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isTraining}>
              <SelectTrigger id="model-select">
                <SelectValue placeholder="Choose a model to train..." />
              </SelectTrigger>
              <SelectContent>
                {MODELS.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div>
                      <p className="font-medium">{model.name}</p>
                      <p className="text-xs text-gray-500">{model.description}</p>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedModel && (
              <p className="text-xs text-gray-600">
                {MODELS.find(m => m.id === selectedModel)?.description}
              </p>
            )}
          </div>

          {/* Dataset Selection */}
          <div className="space-y-2">
            <Label htmlFor="dataset-select">Select Dataset</Label>
            {csvFiles.length === 0 ? (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3">
                <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-800">
                  <p className="font-medium mb-1">No datasets available</p>
                  <p>Upload a CSV file first in the Data Entry section</p>
                </div>
              </div>
            ) : (
              <Select value={selectedDataset} onValueChange={setSelectedDataset} disabled={isTraining}>
                <SelectTrigger id="dataset-select">
                  <SelectValue placeholder="Choose a dataset..." />
                </SelectTrigger>
                <SelectContent>
                  {csvFiles.map((file) => (
                    <SelectItem key={file} value={file}>
                      {file}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* Training Controls */}
          <div className="flex items-center gap-4">
            {!isTraining ? (
              <Button 
                onClick={startTraining} 
                className="flex items-center gap-2"
                disabled={!selectedModel || !selectedDataset || csvFiles.length === 0}
              >
                <Play className="w-4 h-4" />
                Start Training
              </Button>
            ) : (
              <Button onClick={stopTraining} variant="destructive" className="flex items-center gap-2">
                <Square className="w-4 h-4" />
                Stop Training
              </Button>
            )}
            
            {isTraining && (
              <Badge variant="secondary" className="animate-pulse">
                Training in progress...
              </Badge>
            )}
          </div>

          {/* Progress */}
          {isTraining && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Training Progress</span>
                <span>{Math.round(trainingProgress)}%</span>
              </div>
              <Progress value={trainingProgress} className="w-full" />
            </div>
          )}

          {/* Results */}
          {trainingJob && !isTraining && trainingJob.status === 'completed' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 space-y-2">
              <h4 className="font-medium text-green-900">Training Completed</h4>
              <div className="grid grid-cols-2 gap-4 text-sm text-green-800">
                <div>
                  <p className="text-xs text-green-700">Model</p>
                  <p className="font-medium">{trainingJob.results?.model || selectedModel}</p>
                </div>
                <div>
                  <p className="text-xs text-green-700">Accuracy</p>
                  <p className="font-medium">{(trainingJob.results?.accuracy * 100).toFixed(2)}%</p>
                </div>
              </div>
            </div>
          )}

          {/* Configuration Info */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium mb-3">Training Configuration</h4>
            <div className="grid grid-cols-2 gap-4 text-xs text-gray-600">
              <div>
                <p><strong>Model:</strong> {selectedModel ? MODELS.find(m => m.id === selectedModel)?.name : 'Not selected'}</p>
                <p><strong>Dataset:</strong> {selectedDataset || 'Not selected'}</p>
              </div>
              <div>
                <p><strong>Status:</strong> {isTraining ? 'Training...' : 'Ready'}</p>
                <p><strong>Progress:</strong> {Math.round(trainingProgress)}%</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
