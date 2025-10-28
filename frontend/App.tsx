import React, { useState } from 'react';
import { LoginPage } from './components/LoginPage';
import { AdminDashboard } from './components/AdminDashboard';
import { UserDashboard } from './components/UserDashboard';
import { DataEntryForm } from './components/DataEntryForm';
import { PredictionsPage } from './components/PredictionsPage';
import { Toaster } from './components/ui/sonner';

type UserRole = 'admin' | 'user' | null;
type UserPage = 'dashboard' | 'form' | 'predictions';

interface User {
  role: UserRole;
  username: string;
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [currentPage, setCurrentPage] = useState<UserPage>('dashboard');

  const handleLogin = (role: UserRole, username: string) => {
    setUser({ role, username });
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setUser(null);
    setCurrentPage('dashboard');
  };

  const handleNavigate = (page: UserPage) => {
    setCurrentPage(page);
  };

  const handleBackToDashboard = () => {
    setCurrentPage('dashboard');
  };

  // Not logged in - show login page
  if (!user) {
    return (
      <>
        <LoginPage onLogin={handleLogin} />
        <Toaster />
      </>
    );
  }

  // Admin user
  if (user.role === 'admin') {
    return (
      <>
        <AdminDashboard onLogout={handleLogout} username={user.username} />
        <Toaster />
      </>
    );
  }

  // Regular user - different pages based on current selection
  if (currentPage === 'form') {
    return (
      <>
        <DataEntryForm onBack={handleBackToDashboard} />
        <Toaster />
      </>
    );
  }

  if (currentPage === 'predictions') {
    return (
      <>
        <PredictionsPage onBack={handleBackToDashboard} />
        <Toaster />
      </>
    );
  }

  // Default: User dashboard
  return (
    <>
      <UserDashboard 
        onLogout={handleLogout} 
        onNavigate={handleNavigate} 
        username={user.username} 
      />
      <Toaster />
    </>
  );
}