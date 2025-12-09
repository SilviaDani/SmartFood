/**
 * API Configuration
 * Supporta sia sviluppo locale che Docker
 */

// Usa la variabile d'ambiente se disponibile (Docker), altrimenti usa localhost (sviluppo)
const API_BASE_URL = 
  import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  auth: {
    login: `${API_BASE_URL}/api/auth/login`,
    logout: `${API_BASE_URL}/api/auth/logout`,
  },
  data: {
    list: `${API_BASE_URL}/api/data`,
    create: `${API_BASE_URL}/api/data/create`,
    update: `${API_BASE_URL}/api/data/:id`,
  },
  predictions: {
    list: `${API_BASE_URL}/api/predictions`,
    generate: `${API_BASE_URL}/api/predictions/generate`,
  },
};

/**
 * Funzione helper per le richieste API
 */
export async function fetchAPI(
  endpoint: string,
  options: RequestInit = {}
) {
  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  try {
    const response = await fetch(endpoint, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}

/**
 * Esempi di utilizzo nei tuoi componenti:
 * 
 * import { API_ENDPOINTS, fetchAPI } from '@/lib/api';
 * 
 * // GET
 * const data = await fetchAPI(API_ENDPOINTS.data.list);
 * 
 * // POST
 * const result = await fetchAPI(API_ENDPOINTS.data.create, {
 *   method: 'POST',
 *   body: JSON.stringify({ ... })
 * });
 */
