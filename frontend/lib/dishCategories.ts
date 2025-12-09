/**
 * Dish Categories Configuration
 * 
 * Questo file contiene la lista delle macrocategorie di piatti.
 * Puoi facilmente aggiungere, modificare o rimuovere categorie qui.
 * 
 * Esempio di utilizzo nel componente:
 * import { DISH_CATEGORIES } from '../lib/dishCategories';
 */

import { id } from "date-fns/locale";

export const DISH_CATEGORIES = [
  {id: 'carne bianca', label: 'Carne Bianca'},
  {id: 'carne rossa', label: 'Carne Rossa'},
  {id: 'cereali', label: 'Cereali'},
  {id: 'dolci', label: 'Dolci'},
  {id: 'frutta', label: 'Frutta'},
  {id: 'latticini', label: 'Latticini'},
  {id: 'legumi', label: 'Legumi'},
  {id: 'panificati', label: 'Panificati'},
  {id: 'pasta', label: 'Pasta'},
  {id: 'pesce', label: 'Pesce'},
  {id: 'riso', label: 'Riso'},
  {id: 'salse', label: 'Salse'},
  {id: 'uova', label: 'Uova'},
  {id: 'varia', label: 'Varia'},
  {id: 'verdura', label: 'Verdura'},
] as const;

export type DishCategory = typeof DISH_CATEGORIES[number]['id'];

/**
 * Funzione helper per ottenere l'etichetta di una categoria
 */
export function getCategoryLabel(categoryId: string): string {
  const category = DISH_CATEGORIES.find(cat => cat.id === categoryId);
  return category ? category.label : categoryId;
}
