import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Brain, Play, X, AlertCircle, Loader } from 'lucide-react';
import { toast } from 'sonner';

interface AITrainingProps {
  onLogout?: () => void;
}

interface TrainingJobState {
  job_id: string;
  model_id: string;
  dataset_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_step: string;
  accuracy?: number;
  loss?: number;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  pollInterval?: NodeJS.Timer;
}

const MODELS = [
  { id: 'moment', name: 'MOMENT', description: 'Multi-horizon time series forecasting' },
  { id: 'chronos', name: 'Chronos', description: 'Probabilistic time series forecasting' },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'pending':
      return 'bg-blue-50 border-blue-200 text-blue-900';
    case 'running':
      return 'bg-purple-50 border-purple-200 text-purple-900';
    case 'completed':
      return 'bg-green-50 border-green-200 text-green-900';
    case 'failed':
      return 'bg-red-50 border-red-200 text-red-900';
    case 'cancelled':
      return 'bg-gray-50 border-gray-200 text-gray-900';
    default:
      return 'bg-gray-50 border-gray-200 text-gray-900';
  }
};

const getStatusBadgeVariant = (status: string) => {
  switch (status) {
    case 'running':
      return 'secondary';
    case 'completed':
      return 'default';
    case 'failed':
      return 'destructive';
    case 'cancelled':
      return 'outline';
    default:
      return 'outline';
  }
};

export function AITraining({ onLogout }: AITrainingProps) {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [csvFiles, setCsvFiles] = useState<string[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJobState[]>([]);

  // Carica la lista dei file CSV disponibili
  useEffect(() => {
    loadAvailableDatasets();
  }, []);

  // Cleanup polling intervals quando il componente si smonta
  useEffect(() => {
    return () => {
      trainingJobs.forEach(job => {
        if (job.pollInterval) clearInterval(job.pollInterval);
      });
    };
  }, [trainingJobs]);

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
      
      // Crea il nuovo job
      const newJob: TrainingJobState = {
        job_id: data.job_id,
        model_id: selectedModel,
        dataset_id: selectedDataset,
        status: 'pending',
        progress: 0,
        current_step: 'Initialization',
      };
      
      setTrainingJobs([...trainingJobs, newJob]);
      toast.success(`Training started: ${selectedModel}`);
      
      // Inizia il polling per questo job
      pollTrainingProgress(data.job_id);
    } catch (error) {
      console.error('Training error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to start training');
    }
  };

  const pollTrainingProgress = (jobId: string) => {
    let pollAttempts = 0;
    const maxPollAttempts = 300; // 10 minuti al polling di 2 secondi
    let lastShownStatus: string | null = null;

    const pollInterval = setInterval(async () => {
      pollAttempts++;
      
      // Timeout di sicurezza dopo 10 minuti
      if (pollAttempts > maxPollAttempts) {
        console.warn(`Poll timeout for job ${jobId} after ${maxPollAttempts} attempts`);
        clearInterval(pollInterval);
        return;
      }

      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/train/${jobId}/status`);
        
        if (!response.ok) {
          // Continua il polling anche se riceve errore (il server potrebbe essere temporaneamente giù)
          console.warn(`Poll failed with status ${response.status} for job ${jobId}`);
          return;
        }

        const data = await response.json();
        
        setTrainingJobs(prevJobs => 
          prevJobs.map(job => 
            job.job_id === jobId 
              ? {
                  ...job,
                  status: data.status,
                  progress: data.progress || 0,
                  current_step: data.current_step || '',
                  accuracy: data.accuracy,
                  loss: data.loss,
                  error_message: data.error_message,
                  completed_at: data.completed_at,
                }
              : job
          )
        );

        // Ferma il polling se il job è completato o fallito
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          clearInterval(pollInterval);
          
          // Mostra notifica solo se è una transizione di stato (non è la stessa notifica ripetuta)
          if (data.status !== lastShownStatus) {
            lastShownStatus = data.status;
            
            if (data.status === 'completed') {
              toast.success(`Training completed! Accuracy: ${(data.accuracy * 100).toFixed(2)}%`);
            } else if (data.status === 'failed') {
              toast.error(`Training failed: ${data.error_message || 'Unknown error'}`);
            } else if (data.status === 'cancelled') {
              toast.info('Training cancelled');
            }
          }
        }
      } catch (error) {
        // Continua il polling anche in caso di errore di connessione
        console.debug(`Poll connection error for job ${jobId}:`, error);
      }
    }, 2000); // Poll every 2 seconds

    // Salva il pollInterval nel job per poterlo cancellare dopo
    setTrainingJobs(prevJobs => 
      prevJobs.map(job => 
        job.job_id === jobId 
          ? { ...job, pollInterval }
          : job
      )
    );
  };

  const cancelTraining = async (jobId: string) => {
    // Ferma il polling e rimuovi dalla lista IMMEDIATAMENTE
    const job = trainingJobs.find(j => j.job_id === jobId);
    if (job?.pollInterval) {
      clearInterval(job.pollInterval);
    }

    // Rimuovi dalla lista subito
    setTrainingJobs(trainingJobs.filter(j => j.job_id !== jobId));
    
    toast.success('Training cancelled and removed');

    // Invia la richiesta di cancellazione al backend in background (non attendere)
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      fetch(`${apiUrl}/api/train/${jobId}/cancel`, {
        method: 'POST',
      }).catch(error => {
        console.debug(`Background cancel request failed for ${jobId}:`, error);
      });
    } catch (error) {
      console.debug(`Cancel background request error:`, error);
    }
  };

  const removeJobFromList = (jobId: string) => {
    // Ferma il polling se ancora attivo
    const job = trainingJobs.find(j => j.job_id === jobId);
    if (job?.pollInterval) {
      clearInterval(job.pollInterval);
    }

    setTrainingJobs(trainingJobs.filter(j => j.job_id !== jobId));
    toast.success('Training removed from list');
  };

  return (
    <div className="space-y-6">
      {/* Training Control Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Addestramento Modello IA
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label htmlFor="model-select">Seleziona Modello</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger id="model-select">
                <SelectValue placeholder="Scegli un modello da addestrare..." />
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
            <Label htmlFor="dataset-select">Seleziona Dataset</Label>
            {csvFiles.length === 0 ? (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex gap-3">
                <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-800">
                  <p className="font-medium mb-1">Nessun dataset disponibile</p>
                  <p>Carica prima un file CSV nella sezione Inserimento Dati</p>
                </div>
              </div>
            ) : (
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger id="dataset-select">
                  <SelectValue placeholder="Scegli un dataset..." />
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

          {/* Start Training Button */}
          <div className="flex items-center gap-4">
            <Button 
              onClick={startTraining}
              className="flex items-center gap-2"
              disabled={!selectedModel || !selectedDataset || csvFiles.length === 0}
            >
              <Play className="w-4 h-4" />
              Inizia Addestramento
            </Button>
            
            {trainingJobs.length > 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                {trainingJobs.filter(j => j.status === 'running' || j.status === 'pending').length} in corso
              </div>
            )}
          </div>

          {/* Configuration Info */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium mb-3">Configurazione Addestramento</h4>
            <div className="grid grid-cols-2 gap-4 text-xs text-gray-600">
              <div>
                <p><strong>Modello:</strong> {selectedModel ? MODELS.find(m => m.id === selectedModel)?.name : 'Non selezionato'}</p>
                <p><strong>Dataset:</strong> {selectedDataset || 'Non selezionato'}</p>
              </div>
              <div>
                <p><strong>Training Attivi:</strong> {trainingJobs.filter(j => j.status === 'running' || j.status === 'pending').length}</p>
                <p><strong>Completati:</strong> {trainingJobs.filter(j => j.status === 'completed').length}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Training Jobs List */}
      {trainingJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Loader className="w-5 h-5 animate-spin" />
              Training in Corso ({trainingJobs.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainingJobs.map((job, index) => (
              <div key={job.job_id} className={`border rounded-lg p-4 space-y-3 ${getStatusColor(job.status)}`}>
                {/* Header con Info Principali */}
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        #{index + 1}
                      </span>
                      <h4 className="font-medium">
                        {MODELS.find(m => m.id === job.model_id)?.name}
                      </h4>
                      <Badge variant={getStatusBadgeVariant(job.status)}>
                        {job.status === 'running' && (
                          <div className="flex items-center gap-1">
                            <div className="w-1.5 h-1.5 bg-current rounded-full animate-pulse" />
                            {job.status}
                          </div>
                        )}
                        {job.status !== 'running' && job.status}
                      </Badge>
                    </div>
                    <p className="text-sm opacity-75">{job.dataset_id}</p>
                  </div>
                  
                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    {(job.status === 'pending' || job.status === 'running') && (
                      <Button
                        onClick={() => cancelTraining(job.job_id)}
                        variant="ghost"
                        size="sm"
                        className="text-red-600 hover:text-red-800 hover:bg-red-50"
                        title="Cancel this training"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    )}
                    {(job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') && (
                      <Button
                        onClick={() => removeJobFromList(job.job_id)}
                        variant="ghost"
                        size="sm"
                        className="text-gray-600 hover:text-gray-800 hover:bg-gray-200"
                        title="Remove from list"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                </div>

                {/* Progress Bar (solo se running o pending) */}
                {(job.status === 'running' || job.status === 'pending') && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-medium">
                      <span>{job.current_step || 'Initializing...'}</span>
                      <span>{Math.round(job.progress)}%</span>
                    </div>
                    <Progress value={job.progress} className="w-full" />
                  </div>
                )}

                {/* Results (completed) */}
                {job.status === 'completed' && (
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="opacity-75 text-xs">Accuratezza</p>
                      <p className="font-semibold text-lg">{(job.accuracy! * 100).toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="opacity-75 text-xs">Loss</p>
                      <p className="font-semibold text-lg">{job.loss!.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="opacity-75 text-xs">Stato</p>
                      <p className="font-semibold">✓ Completato</p>
                    </div>
                  </div>
                )}

                {/* Error Message (failed) */}
                {job.status === 'failed' && (
                  <div className="bg-red-100 border border-red-300 rounded p-3 text-sm">
                    <p className="font-medium mb-1">Errore:</p>
                    <p className="text-xs">{job.error_message || 'Unknown error'}</p>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
