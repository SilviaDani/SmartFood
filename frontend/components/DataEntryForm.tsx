import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Calendar } from './ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { CalendarIcon, ArrowLeft, Save, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';
import { toast } from 'sonner@2.0.3';

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
    dishName: '',
    portionsPrepared: '',
    portionsWasted: ''
  });
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validation
    if (!formData.school || !formData.dishName || !formData.portionsPrepared) {
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
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button onClick={onBack} variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          <h1 className="text-3xl text-gray-900">Enter Meal Data</h1>
          <p className="text-gray-600 mt-2">Record information about prepared meals and food waste</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Meal Information</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* School Selection */}
              <div className="space-y-2">
                <Label htmlFor="school">School *</Label>
                <Select value={formData.school} onValueChange={(value) => setFormData({...formData, school: value})}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a school..." />
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
                <Label>Date *</Label>
                <Popover open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
                  <PopoverTrigger asChild>
                    <Button variant="outline" className="w-full justify-start text-left">
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {format(formData.date, 'PPP')}
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
                    />
                  </PopoverContent>
                </Popover>
              </div>

              {/* Dish Name */}
              <div className="space-y-2">
                <Label htmlFor="dishName">Dish Name *</Label>
                <Input
                  id="dishName"
                  type="text"
                  value={formData.dishName}
                  onChange={(e) => setFormData({...formData, dishName: e.target.value})}
                  placeholder="e.g., Spaghetti with Marinara Sauce"
                  required
                />
              </div>

              {/* Portions Prepared */}
              <div className="space-y-2">
                <Label htmlFor="portionsPrepared">Portions Prepared *</Label>
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
                <Label htmlFor="portionsWasted">Portions Wasted</Label>
                <Input
                  id="portionsWasted"
                  type="number"
                  min="0"
                  max={formData.portionsPrepared || undefined}
                  value={formData.portionsWasted}
                  onChange={(e) => setFormData({...formData, portionsWasted: e.target.value})}
                  placeholder="15"
                />
                <p className="text-xs text-gray-500">Leave empty if no waste occurred</p>
              </div>

              {/* Waste Summary */}
              {formData.portionsPrepared && formData.portionsWasted && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertCircle className="w-4 h-4 text-gray-500" />
                    <span className="text-sm text-gray-700">Waste Summary</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Portions Served:</p>
                      <p className="text-lg">{parseInt(formData.portionsPrepared) - parseInt(formData.portionsWasted)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Waste Percentage:</p>
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
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Save Entry
                    </>
                  )}
                </Button>
                <Button type="button" variant="outline" onClick={onBack}>
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Help Text */}
        <div className="mt-8 text-center text-sm text-gray-600">
          <p>Make sure to enter accurate data to help improve prediction models.</p>
          <p>All fields marked with * are required.</p>
        </div>
      </div>
    </div>
  );
}