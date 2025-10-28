import React, { useState } from 'react';
import { ReactNode } from "react";
import { Calendar } from './ui/calendar';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { CalendarIcon, ArrowLeft } from 'lucide-react';
import { format } from 'date-fns';

interface PredictionsPageProps {
  onBack: () => void;
}

// Reusing the same data structure from the original App
const schools = [
  { id: '1', name: 'Lincoln Elementary School' },
  { id: '2', name: 'Washington Middle School' },
  { id: '3', name: 'Roosevelt High School' },
  { id: '4', name: 'Jefferson Primary School' },
  { id: '5', name: 'Kennedy Secondary School' }
];

const menuData = {
  '1': { // Lincoln Elementary
    recipes: [
      { id: '1', name: 'Spaghetti with Marinara Sauce', portions: 85, confidence: 92 },
      { id: '2', name: 'Chicken Nuggets with Sweet Potato Fries', portions: 120, confidence: 88 },
      { id: '3', name: 'Turkey and Cheese Sandwich', portions: 65, confidence: 95 },
      { id: '4', name: 'Vegetarian Pizza Slice', portions: 45, confidence: 78 }
    ]
  },
  '2': { // Washington Middle
    recipes: [
      { id: '1', name: 'BBQ Pulled Pork Sandwich', portions: 150, confidence: 85 },
      { id: '2', name: 'Caesar Salad with Grilled Chicken', portions: 95, confidence: 90 },
      { id: '3', name: 'Beef Tacos with Rice and Beans', portions: 180, confidence: 93 },
      { id: '4', name: 'Margherita Pizza', portions: 75, confidence: 82 }
    ]
  },
  '3': { // Roosevelt High
    recipes: [
      { id: '1', name: 'Grilled Chicken Breast with Quinoa', portions: 200, confidence: 87 },
      { id: '2', name: 'Beef Burger with Fries', portions: 250, confidence: 91 },
      { id: '3', name: 'Asian Stir-Fry with Tofu', portions: 110, confidence: 79 },
      { id: '4', name: 'Mediterranean Wrap', portions: 130, confidence: 85 },
      { id: '5', name: 'Fish and Chips', portions: 160, confidence: 88 }
    ]
  }
};

export function PredictionsPage({ onBack }: PredictionsPageProps) {
  const [selectedSchool, setSelectedSchool] = useState<string>('');
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'bg-green-100 text-green-800';
    if (confidence >= 80) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const currentMenu = selectedSchool ? menuData[selectedSchool as keyof typeof menuData] : null;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <Button onClick={onBack} variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          <div className="text-center space-y-2">
            <h1 className="text-3xl text-gray-900">Meal Predictions</h1>
            <p className="text-gray-600">Select a school and date to view AI-powered meal predictions</p>
          </div>
        </div>

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          {/* School Selector */}
          <div className="space-y-2">
            <label className="block text-sm text-gray-700">Select School</label>
            <Select value={selectedSchool} onValueChange={setSelectedSchool}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Choose a school..." />
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

          {/* Date Picker */}
          <div className="space-y-2">
            <label className="block text-sm text-gray-700">Select Date</label>
            <Popover open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-start text-left"
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {format(selectedDate, 'PPP')}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="single"
                  selected={selectedDate}
                  onSelect={(date) => {
                    if (date) {
                      setSelectedDate(date);
                      setIsCalendarOpen(false);
                    }
                  }}
                  initialFocus
                />
              </PopoverContent>
            </Popover>
          </div>
        </div>

        {/* Menu Display */}
        {selectedSchool && currentMenu && (
          <div className="space-y-6">
            <div className="text-center">
              <h2 className="text-2xl text-gray-900">
                Predictions for {schools.find(s => s.id === selectedSchool)?.name}
              </h2>
              <p className="text-gray-600 mt-1">
                {format(selectedDate, 'EEEE, MMMM do, yyyy')}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {currentMenu.recipes.map((recipe) => (
                <Card key={recipe.id} className="hover:shadow-lg transition-shadow duration-200">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg leading-tight">{recipe.name}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Predicted Portions:</span>
                      <span className="text-lg">{recipe.portions}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Confidence:</span>
                      <Badge 
                        variant="secondary" 
                        className={getConfidenceColor(recipe.confidence)}
                      >
                        {recipe.confidence}%
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Additional Info */}
            <div className="bg-blue-50 p-6 rounded-lg">
              <h3 className="text-lg text-blue-900 mb-3">How to Use These Predictions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
                <div>
                  <p><strong>High Confidence (90%+):</strong> Very reliable predictions based on historical data</p>
                  <p><strong>Medium Confidence (80-89%):</strong> Good predictions with some variability expected</p>
                </div>
                <div>
                  <p><strong>Lower Confidence ({'<'}80%):</strong> Use as guidance but prepare for higher variance</p>
                  <p><strong>Tip:</strong> Consider weather, special events, and holidays that may affect demand</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!selectedSchool && (
          <div className="text-center py-12">
            <div className="max-w-md mx-auto">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <CalendarIcon className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg text-gray-900 mb-2">Select a School to Begin</h3>
              <p className="text-gray-600">
                Choose a school from the dropdown above to view AI-powered meal predictions for your selected date.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}