import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Calendar } from './ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { CalendarIcon, ArrowLeft, Save, AlertCircle, Upload } from 'lucide-react';
import { format } from 'date-fns';
import { it } from 'date-fns/locale';
import { toast } from 'sonner';
import { CSVUploader } from './CSVUploader';
import { DISH_CATEGORIES } from '../lib/dishCategories';

interface DataEntryFormProps {
  onBack: () => void;
}

const schools = [
  { id: '1', name: 'Lincoln Elementary School' },
  { id: '2', name: 'Washington Middle School' },
  { id: '3', name: 'Roosevelt High School' },
  { id: '4', name: 'Jefferson Primary School' },
  { id: '5', name: 'Kennedy Secondary School' }
];

export function DataEntryForm({ onBack }: DataEntryFormProps) {
  const [formData, setFormData] = useState({
    school: '',
    date: new Date(),
    category: '',
    dishName: '',
    portionsPrepared: '',
    portionsWasted: ''
  });
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validation
    if (!formData.school || !formData.category || !formData.dishName || !formData.portionsPrepared) {
      toast.error('Please fill in all required fields');
      return;
    }

    const prepared = parseInt(formData.portionsPrepared);
    const wasted = parseInt(formData.portionsWasted) || 0;

    if (prepared <= 0) {
      toast.error('Portions prepared must be greater than 0');
      return;
    }

    if (wasted < 0) {
      toast.error('Portions wasted cannot be negative');
      return;
    }

    if (wasted > prepared) {
      toast.error('Portions wasted cannot exceed portions prepared');
      return;
    }

    setIsSubmitting(true);

    // Simulate API call
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast.success('Data entry saved successfully!');
      
      // Reset form
      setFormData({
        school: '',
        date: new Date(),
        category: '',
        dishName: '',
        portionsPrepared: '',
        portionsWasted: ''
      });
    } catch (error) {
      toast.error('Failed to save data. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const wastePercentage = formData.portionsPrepared && formData.portionsWasted 
    ? Math.round((parseInt(formData.portionsWasted) / parseInt(formData.portionsPrepared)) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button onClick={onBack} variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Torna al Pannello
          </Button>
          <h1 className="text-3xl text-gray-900">Inserisci i Dati del Pasto</h1>
          <p className="text-gray-600 mt-2">Registra informazioni sui pasti preparati e sulle porzioni sprecate</p>
        </div>

        {/* Tabs Container */}
        <Tabs defaultValue="single" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="single" className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Singolo Inserimento
            </TabsTrigger>
            <TabsTrigger value="csv" className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Caricamento in massa
            </TabsTrigger>
          </TabsList>

          {/* Tab 1: Single Entry */}
          <TabsContent value="single" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Informazioni Pasto</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
              {/* School Selection */}
              <div className="space-y-2">
                <Label htmlFor="school">Scuola *</Label>
                <Select value={formData.school} onValueChange={(value) => setFormData({...formData, school: value})}>
                  <SelectTrigger>
                    <SelectValue placeholder="Seleziona una scuola..." />
                  </SelectTrigger>
                  <SelectContent>
                    {schools.map((school) => (
                      <SelectItem key={school.id} value={school.id}>
                        {school.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Date Selection */}
              <div className="space-y-2">
                <Label>Data *</Label>
                <Popover open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className="w-full justify-start text-left">
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {format(formData.date, 'PPP', { locale: it })}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={formData.date}
                      onSelect={(date) => {
                        if (date) {
                          setFormData({...formData, date});
                          setIsCalendarOpen(false);
                        }
                      }}
                      initialFocus
                      locale={it}
                    />
                  </PopoverContent>
                </Popover>
              </div>

              {/* Dish Category */}
              <div className="space-y-2">
                <Label htmlFor="category">Categoria Piatto *</Label>
                <Select value={formData.category} onValueChange={(value) => setFormData({...formData, category: value})}>
                  <SelectTrigger>
                    <SelectValue placeholder="Seleziona una categoria..." />
                  </SelectTrigger>
                  <SelectContent style={{ backgroundColor: '#e5e7eb', color: '#1f2937' }}>
                    {DISH_CATEGORIES.map((cat) => (
                      <SelectItem key={cat.id} value={cat.id}>
                        {cat.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="dishName">Nome del Piatto *</Label>
                <Input
                  id="dishName"
                  type="text"
                  value={formData.dishName}
                  onChange={(e) => setFormData({...formData, dishName: e.target.value})}
                  placeholder="es. Spaghetti con Salsa di Pomodoro"
                  required
                />
              </div>

              {/* Portions Prepared */}
              <div className="space-y-2">
                <Label htmlFor="portionsPrepared">Porzioni Preparate *</Label>
                <Input
                  id="portionsPrepared"
                  type="number"
                  min="1"
                  value={formData.portionsPrepared}
                  onChange={(e) => setFormData({...formData, portionsPrepared: e.target.value})}
                  placeholder="120"
                  required
                />
              </div>

              {/* Portions Wasted */}
              <div className="space-y-2">
                <Label htmlFor="portionsWasted">Porzioni Sprecate</Label>
                <Input
                  id="portionsWasted"
                  type="number"
                  min="0"
                  max={formData.portionsPrepared || undefined}
                  value={formData.portionsWasted}
                  onChange={(e) => setFormData({...formData, portionsWasted: e.target.value})}
                  placeholder="15"
                />
                <p className="text-xs text-gray-500">Lascia vuoto se non c'è stato spreco</p>
              </div>

              {/* Waste Summary */}
              {formData.portionsPrepared && formData.portionsWasted && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertCircle className="w-4 h-4 text-gray-500" />
                    <span className="text-sm text-gray-700">Riepilogo Spreco</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Porzioni Servite:</p>
                      <p className="text-lg">{parseInt(formData.portionsPrepared) - parseInt(formData.portionsWasted)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Percentuale Spreco:</p>
                      <p className={`text-lg ${wastePercentage > 15 ? 'text-red-600' : wastePercentage > 10 ? 'text-orange-600' : 'text-green-600'}`}>
                        {wastePercentage}%
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <div className="flex gap-4 pt-4">
                <Button 
                  type="submit" 
                  disabled={isSubmitting}
                  className="flex-1"
                >
                  {isSubmitting ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Salvataggio in corso...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Salva Inserimento
                    </>
                  )}
                </Button>
                <Button type="button" variant="outline" onClick={onBack}>
                  Annulla
                </Button>
              </div>
            </form>
          </CardContent>
            </Card>

            {/* Help Text */}
            <div className="text-center text-sm text-gray-600">
              <p>Assicurati di inserire i dati accurati per aiutare a migliorare i modelli di previsione.</p>
              <p>Tutti i campi contrassegnati con * sono obbligatori.</p>
            </div>
          </TabsContent>

          {/* Tab 2: Bulk Upload */}
          <TabsContent value="csv" className="space-y-6">
            <CSVUploader />

            {/* Help Text */}
            <div className="text-center text-sm text-gray-600">
              <p>Carica più inserimenti contemporaneamente usando un file CSV.</p>
              <p>Questo è il modo più veloce per importare grandi quantità di dati.</p>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}