import React, { useState, useEffect } from 'react';
import { ArrowLeft, Loader, TrendingUp, Calendar as CalendarIcon } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { DateRangeCalendar } from './DateRangeCalendar';
import { countWorkingDays } from '../lib/dateUtils';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PredictionsPageProps {
  onBack: () => void;
}

interface Prediction {
  date: string;
  portions: number;
  confidence: number;
}

interface PredictionResponse {
  success: boolean;
  school: string;
  model: string;
  forecast_days: number;
  predictions: Prediction[];
  message?: string;
}

export function PredictionsPage({ onBack }: PredictionsPageProps) {
  // State per i controlli
  const [schools, setSchools] = useState<string[]>([]);
  const [selectedSchool, setSelectedSchool] = useState<string>('');
  const [dishes, setDishes] = useState<string[]>([]);
  const [selectedDish, setSelectedDish] = useState<string>('');
  const [loadingDishes, setLoadingDishes] = useState(false);
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('chronos');
  const [loading, setLoading] = useState(false);
  const [loadingSchools, setLoadingSchools] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // State per i risultati
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [responseData, setResponseData] = useState<PredictionResponse | null>(null);

  // Carica la lista delle scuole all'avvio
  useEffect(() => {
    loadSchools();
  }, []);

  // Carica i piatti quando cambia la scuola selezionata
  useEffect(() => {
    if (selectedSchool) {
      loadDishes(selectedSchool);
    }
  }, [selectedSchool]);

  const loadSchools = async () => {
    try {
      setLoadingSchools(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/schools`);
      
      if (!response.ok) {
        throw new Error('Failed to load schools');
      }
      
      const data = await response.json();
      
      if (data.success && Array.isArray(data.schools)) {
        setSchools(data.schools);
        if (data.schools.length > 0) {
          setSelectedSchool(data.schools[0]);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load schools');
      setSchools([]);
    } finally {
      setLoadingSchools(false);
    }
  };

  const loadDishes = async (school: string) => {
    try {
      setLoadingDishes(true);
      setDishes([]);
      setSelectedDish('');
      
      const response = await fetch(`${API_BASE_URL}/api/dishes/${encodeURIComponent(school)}`);
      
      if (!response.ok) {
        throw new Error('Failed to load dishes');
      }
      
      const data = await response.json();
      
      if (data.success && Array.isArray(data.dishes)) {
        setDishes(data.dishes);
        if (data.dishes.length > 0) {
          setSelectedDish(data.dishes[0]);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Error loading dishes:', err);
      setDishes([]);
    } finally {
      setLoadingDishes(false);
    }
  };

  const handleGeneratePredictions = async () => {
    if (!selectedSchool) {
      setError('Please select a school');
      return;
    }

    if (!startDate || !endDate) {
      setError('Please select both start and end dates');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setPredictions([]);

      const payload = {
        school: selectedSchool,
        model: selectedModel,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0],
        dish_name: selectedDish || undefined,
      };

      console.log('📤 Sending prediction request:', payload);

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data: PredictionResponse = await response.json();

      console.log('📥 Received response:', data);

      if (!data.success) {
        setError(data.message || 'Failed to generate predictions');
        return;
      }

      setPredictions(data.predictions);
      setResponseData(data);
      setGeneratedAt(new Date().toLocaleString());
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to generate predictions';
      console.error('❌ Error:', errorMsg);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'bg-green-100 text-green-800';
    if (confidence >= 0.8) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getConfidencePercentage = (confidence: number) => {
    return Math.round(confidence * 100);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <Button onClick={onBack} variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2">
              <TrendingUp className="w-8 h-8 text-blue-600" />
              <h1 className="text-3xl text-gray-900">Meal Predictions</h1>
            </div>
            <p className="text-gray-600">Generate AI-powered predictions for meal portions at your selected school</p>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
            <p className="font-semibold">Error</p>
            <p>{error}</p>
          </div>
        )}

        {/* Controls */}
        <Card className="border-2 bg-white shadow-lg">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 border-b">
            <CardTitle className="text-2xl">⚙️ Prediction Settings</CardTitle>
            <p className="text-sm text-gray-600 mt-1">Configure your forecast parameters</p>
          </CardHeader>
          <CardContent className="space-y-6 pt-8">
            {loadingSchools ? (
              <div className="flex items-center justify-center py-12">
                <Loader className="w-6 h-6 animate-spin text-blue-600 mr-2" />
                <span className="text-gray-600">Loading schools...</span>
              </div>
            ) : schools.length === 0 ? (
              <div className="text-center py-8 bg-yellow-50 rounded-lg border border-yellow-200">
                <p className="text-yellow-800 font-semibold">No Schools Available</p>
                <p className="text-sm text-yellow-700 mt-1">Please upload CSV data first to get started</p>
              </div>
            ) : (
              <>
                {/* School Selector */}
                <div className="space-y-2 bg-gray-50 p-4 rounded-lg">
                  <label className="block text-sm font-bold text-gray-800">🏫 School</label>
                  <Select value={selectedSchool} onValueChange={setSelectedSchool}>
                    <SelectTrigger className="w-full border-2 border-gray-300 hover:border-blue-500">
                      <SelectValue placeholder="Choose a school..." />
                    </SelectTrigger>
                    <SelectContent style={{ backgroundColor: '#eff0f2ff', color: '#1f2937' }}>
                      {schools.map((school) => (
                        <SelectItem key={school} value={school}>
                          {school}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Dish Selector */}
                <div className="space-y-2 bg-gray-50 p-4 rounded-lg">
                  <label className="block text-sm font-bold text-gray-800">🍽️ Dish Type</label>
                  {loadingDishes ? (
                    <div className="flex items-center gap-2 py-3 px-4 bg-white rounded border-2 border-gray-300">
                      <Loader className="w-4 h-4 animate-spin text-blue-600" />
                      <span className="text-gray-600 text-sm">Loading dishes...</span>
                    </div>
                  ) : dishes.length === 0 ? (
                    <div className="py-3 px-4 bg-white rounded border-2 border-gray-300 text-gray-500 text-sm">
                      No dishes available for this school
                    </div>
                  ) : (
                    <Select value={selectedDish} onValueChange={setSelectedDish}>
                      <SelectTrigger className="w-full border-2 border-gray-300 hover:border-blue-500">
                        <SelectValue placeholder="Choose a dish..." />
                      </SelectTrigger>
                      <SelectContent style={{ backgroundColor: '#eff0f2ff', color: '#1f2937' }}>
                        {dishes.map((dish) => (
                          <SelectItem key={dish} value={dish}>
                            {dish}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                  <p className="text-xs text-gray-500 mt-2">
                    📌 Leave empty to predict for all dishes combined
                  </p>
                </div>

                {/* Model Selector */}
                                {/* Model Selector */}
                <div className="space-y-2 bg-gray-50 p-4 rounded-lg">
                  <label className="block text-sm font-bold text-gray-800">🤖 AI Model</label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="w-full border-2 border-gray-300 hover:border-blue-500">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent style={{ backgroundColor: '#eff0f2ff', color: '#1f2937' }}>
                      <SelectItem value="chronos">Chronos (Trend-based)</SelectItem>
                      <SelectItem value="moment">MOMENT (Moving Average)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-gray-500 mt-2">
                    {selectedModel === 'chronos'
                      ? '📈 Uses trend analysis to predict future portions'
                      : '📊 Uses moving average of historical data'}
                  </p>
                </div>

                {/* Forecast Days Slider */}
                <div className="space-y-2 bg-gray-50 p-4 rounded-lg">
                  <label className="block text-sm font-bold text-gray-800">📅 Forecast Period</label>
                  <Popover open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
                    <PopoverTrigger asChild>
                      <Button variant="outline" className="w-full justify-start text-left h-auto py-3">
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        <div className="flex flex-col">
                          <span className="text-sm">
                            {startDate && endDate
                              ? `${startDate.toLocaleDateString()} - ${endDate.toLocaleDateString()}`
                              : 'Select date range...'}
                          </span>
                          {startDate && endDate && (
                            <span className="text-xs text-gray-500 mt-1">
                              {countWorkingDays(startDate, endDate)} working days
                            </span>
                          )}
                        </div>
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-fit p-0 border-0 !shadow-lg !rounded-md [&>*]:w-auto" align="start">
                      <DateRangeCalendar
                        startDate={startDate}
                        endDate={endDate}
                        onStartDateChange={setStartDate}
                        onEndDateChange={setEndDate}
                        onConfirm={() => setIsCalendarOpen(false)}
                      />
                    </PopoverContent>
                  </Popover>
                  <p className="text-xs text-gray-500 mt-2">
                    ℹ️ Weekends (Saturday & Sunday) are automatically excluded from predictions
                  </p>
                </div>

                {/* MAIN GENERATE BUTTON */}
                <button
                  onClick={handleGeneratePredictions}
                  disabled={loading}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    fontSize: '16px',
                    fontWeight: '600',
                    backgroundColor: loading ? '#9ca3af' : '#000000',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                  }}
                  onMouseEnter={(e) => {
                    if (!loading) {
                      (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#2563eb';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!loading) {
                      (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#000000';
                    }
                  }}
                >
                  {loading ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" style={{ marginRight: '8px' }} />
                      Generating...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="w-4 h-4" style={{ marginRight: '8px' }} />
                      Generate Predictions
                    </>
                  )}
                </button>

                {/* Helper Text */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                  <p className="text-sm text-blue-900">
                    ✨ Click the button above to generate AI predictions for {selectedSchool}
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        {predictions.length > 0 && (
          <div className="space-y-6">
            {/* Summary */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{selectedSchool}</CardTitle>
                    <p className="text-sm text-gray-600 mt-1">
                      {selectedDish && <span>Dish: <strong>{selectedDish}</strong> • </span>}
                      {selectedModel.toUpperCase()} Model • {startDate && endDate && `${startDate.toLocaleDateString()} to ${endDate.toLocaleDateString()}`}
                    </p>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    Generated: {generatedAt}
                  </div>
                </div>
              </CardHeader>
            </Card>

            {/* Predictions Table */}
            <Card>
              <CardHeader>
                <CardTitle>Forecast Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Date</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Predicted Portions</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Confidence</th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-700">Reliability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((pred, idx) => (
                        <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="py-3 px-4">{new Date(pred.date).toLocaleDateString()}</td>
                          <td className="py-3 px-4">
                            <span className="text-lg font-semibold text-gray-900">{pred.portions}</span>
                            <span className="text-sm text-gray-500 ml-2">portions</span>
                          </td>
                          <td className="py-3 px-4">
                            <Badge
                              variant="secondary"
                              className={getConfidenceColor(pred.confidence)}
                            >
                              {getConfidencePercentage(pred.confidence)}%
                            </Badge>
                          </td>
                          <td className="py-3 px-4">
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  pred.confidence >= 0.9
                                    ? 'bg-green-500'
                                    : pred.confidence >= 0.8
                                    ? 'bg-yellow-500'
                                    : 'bg-red-500'
                                }`}
                                style={{ width: `${pred.confidence * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-gray-600 text-sm mb-2">Average Portions</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {Math.round(
                        predictions.reduce((sum, p) => sum + p.portions, 0) / predictions.length
                      )}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-gray-600 text-sm mb-2">Average Confidence</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {Math.round(
                        (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length) * 100
                      )}
                      %
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-gray-600 text-sm mb-2">Forecast Range</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {Math.min(...predictions.map(p => p.portions))} - {Math.max(...predictions.map(p => p.portions))}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Info Box */}
            <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">Understanding Your Predictions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
                <div>
                  <p><strong>High Confidence (90%+):</strong> Very reliable predictions based on consistent historical patterns</p>
                  <p className="mt-2"><strong>Medium Confidence (80-89%):</strong> Good predictions with some variability expected</p>
                </div>
                <div>
                  <p><strong>Lower Confidence ({`<`}80%):</strong> Use as guidance but prepare for higher variance</p>
                  <p className="mt-2"><strong>Tip:</strong> Consider special events, holidays, and weather that may affect demand</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && predictions.length === 0 && !error && (
          <div className="text-center py-12">
            <div className="max-w-md mx-auto">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                <TrendingUp className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-lg text-gray-900 mb-2">Ready to Forecast</h3>
              <p className="text-gray-600">
                Select a school, choose your AI model, and set the forecast period above to generate predictions.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}