import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { UserCircle, PlusCircle, BarChart3 } from 'lucide-react';

interface UserDashboardProps {
  onLogout: () => void;
  onNavigate: (page: 'form' | 'predictions') => void;
  username: string;
}

export function UserDashboard({ onLogout, onNavigate, username }: UserDashboardProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <UserCircle className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl text-gray-900">Pannello Utente</h1>
                <p className="text-sm text-gray-600">Benvenuto, {username}</p>
              </div>
            </div>
            <Button onClick={onLogout} variant="outline">
              Logout
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-2xl text-gray-900 mb-4">Cosa desideri fare?</h2>
          <p className="text-gray-600">Scegli un'opzione per iniziare</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-2xl mx-auto">
          {/* Data Entry Card */}
          <Card className="hover:shadow-lg transition-shadow duration-200 cursor-pointer" onClick={() => onNavigate('form')}>
            <CardHeader className="text-center pb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                <PlusCircle className="w-8 h-8 text-blue-600" />
              </div>
              <CardTitle className="text-xl">Inserisci Nuovi Dati</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-gray-600 mb-6 leading-relaxed">
                Registra nuovi dati sui pasti includendo informazioni sulla scuola, data, dettagli del piatto, porzioni preparate e quantità di scarto.
              </p>
              <Button onClick={() => onNavigate('form')} className="w-full">
                Aggiungi Nuovo Inserimento
              </Button>
            </CardContent>
          </Card>

          {/* Predictions Card */}
          <Card className="hover:shadow-lg transition-shadow duration-200 cursor-pointer" onClick={() => onNavigate('predictions')}>
            <CardHeader className="text-center pb-4">
              <div className="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
                <BarChart3 className="w-8 h-8 text-green-600" />
              </div>
              <CardTitle className="text-xl">Visualizza Previsioni</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-gray-600 mb-6 leading-relaxed">
                Accedi alle previsioni alimentari alimentate dall'IA per diverse scuole e date per aiutare nella pianificazione dei pasti e nella riduzione degli scarti.
              </p>
              <Button onClick={() => onNavigate('predictions')} variant="outline" className="w-full">
                Visualizza Previsioni
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Quick Stats */}
        <div className="mt-16 pt-8 border-t border-gray-200">
          <h3 className="text-lg text-gray-900 mb-6 text-center">Panoramica Rapida</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-2xl text-blue-600 mb-2">15</div>
              <p className="text-sm text-gray-600">Inserimenti Questa Settimana</p>
            </div>
            <div className="text-center">
              <div className="text-2xl text-green-600 mb-2">94%</div>
              <p className="text-sm text-gray-600">Accuratezza Previsioni</p>
            </div>
            <div className="text-center">
              <div className="text-2xl text-orange-600 mb-2">12%</div>
              <p className="text-sm text-gray-600">Riduzione Scarti Media</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}