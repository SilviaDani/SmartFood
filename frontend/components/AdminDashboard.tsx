import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Brain, Database, Play, Square, BarChart3, Calendar, Users, Shield } from 'lucide-react';
import { AITraining } from './AITraining';

interface AdminDashboardProps {
  onLogout: () => void;
  username: string;
}

// Mock data for admin dashboard
const trainingHistory = [
  { id: 1, date: '2024-10-01', duration: '2h 15m', accuracy: 94.2, status: 'completato' },
  { id: 2, date: '2024-09-28', duration: '1h 45m', accuracy: 92.8, status: 'completato' },
  { id: 3, date: '2024-09-25', duration: '2h 30m', accuracy: 91.5, status: 'completato' },
];

const schoolStats = [
  { school: 'scuola elementare Dante Alighieri', totalMeals: 1250, avgWaste: 12.5, predictions: 45 },
  { school: 'scuola media P. Vasari', totalMeals: 2100, avgWaste: 15.8, predictions: 62 },
  { school: 'liceo Da Vinci', totalMeals: 3200, avgWaste: 18.2, predictions: 78 },
  { school: 'scuola primaria Dante Alighieri', totalMeals: 980, avgWaste: 10.3, predictions: 38 },
  { school: 'scuola secondaria Masaccio', totalMeals: 2800, avgWaste: 16.9, predictions: 71 },
];

const recentEntries = [
  { id: 1, date: '2024-10-02', school: 'scuola elementare Dante Alighieri', dish: 'Spaghetti alla Marinara', prepared: 120, wasted: 15 },
  { id: 2, date: '2024-10-02', school: 'scuola media P. Vasari', dish: 'Petto di pollo', prepared: 180, wasted: 22 },
  { id: 3, date: '2024-10-01', school: 'liceo Da Vinci', dish: 'Burger Vegetale', prepared: 200, wasted: 35 },
  { id: 4, date: '2024-10-01', school: 'scuola primaria Dante Alighieri', dish: 'Pasta al Pesto', prepared: 95, wasted: 8 },
  { id: 5, date: '2024-10-01', school: 'scuola secondaria Masaccio', dish: 'Pesce alla Diavola', prepared: 160, wasted: 28 },
];

export function AdminDashboard({ onLogout, username }: AdminDashboardProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const startTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 100;
        }
        return prev + Math.random() * 10;
      });
    }, 500);
  };

  const stopTraining = () => {
    setIsTraining(false);
    setTrainingProgress(0);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl text-gray-900">Pannello Amministratore</h1>
                <p className="text-sm text-gray-600">Benvenuto, {username}</p>
              </div>
            </div>
            <Button onClick={onLogout} variant="outline">
              Logout
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Panoramica</TabsTrigger>
            <TabsTrigger value="training">Addestramento IA</TabsTrigger>
            <TabsTrigger value="data">Dati della Scuola</TabsTrigger>
            <TabsTrigger value="entries">Inserimenti Recenti</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm">Scuole Totali</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl">5</div>
                  <p className="text-xs text-muted-foreground">Scuole attive</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm">Pasti Totali</CardTitle>
                  <BarChart3 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl">10,330</div>
                  <p className="text-xs text-muted-foreground">Questo mese</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm">Tasso dello Spreco Medio</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl">14.7%</div>
                  <p className="text-xs text-muted-foreground">-2.1% dal mese scorso</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm">Accuratezza Modello</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl">94.2%</div>
                  <p className="text-xs text-muted-foreground">Ultimo addestramento</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Storico Recente dell'Addestramento</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {trainingHistory.map((training) => (
                    <div key={training.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-4">
                        <Calendar className="w-5 h-5 text-gray-400" />
                        <div>
                          <p className="text-sm">{training.date}</p>
                          <p className="text-xs text-gray-500">Durata: {training.duration}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <Badge variant="secondary">{training.accuracy}% accuratezza</Badge>
                        <Badge variant="outline" className="text-green-600">
                          {training.status}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="training" className="space-y-6">
            <AITraining />
          </TabsContent>

          <TabsContent value="data" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Panoramica Prestazioni Scuola</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Scuola</TableHead>
                      <TableHead>Pasti Totali</TableHead>
                      <TableHead>% Spreco Medio</TableHead>
                      <TableHead>Previsioni Effettuate</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {schoolStats.map((school, index) => (
                      <TableRow key={index}>
                        <TableCell>{school.school}</TableCell>
                        <TableCell>{school.totalMeals.toLocaleString()}</TableCell>
                        <TableCell>
                          <Badge 
                            variant={school.avgWaste < 12 ? "secondary" : school.avgWaste < 16 ? "outline" : "destructive"}
                          >
                            {school.avgWaste}%
                          </Badge>
                        </TableCell>
                        <TableCell>{school.predictions}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="entries" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Dati Inseriti Recentemente</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Data</TableHead>
                      <TableHead>Scuola</TableHead>
                      <TableHead>Piatto</TableHead>
                      <TableHead>Preparato</TableHead>
                      <TableHead>Spreco</TableHead>
                      <TableHead>% Sreco</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentEntries.map((entry) => (
                      <TableRow key={entry.id}>
                        <TableCell>{entry.date}</TableCell>
                        <TableCell>{entry.school}</TableCell>
                        <TableCell>{entry.dish}</TableCell>
                        <TableCell>{entry.prepared}</TableCell>
                        <TableCell>{entry.wasted}</TableCell>
                        <TableCell>
                          <Badge 
                            variant={
                              (entry.wasted / entry.prepared * 100) < 10 ? "secondary" : 
                              (entry.wasted / entry.prepared * 100) < 15 ? "outline" : "destructive"
                            }
                          >
                            {Math.round(entry.wasted / entry.prepared * 100)}%
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}