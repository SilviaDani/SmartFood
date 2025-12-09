/**
 * Utility functions for working with dates and weekdays
 */

/**
 * Check if a date is a weekend (Saturday or Sunday)
 */
export function isWeekend(date: Date): boolean {
  const day = date.getDay();
  return day === 0 || day === 6; // 0 = Sunday, 6 = Saturday
}

/**
 * Check if a date is a weekday (Monday-Friday)
 */
export function isWeekday(date: Date): boolean {
  return !isWeekend(date);
}

/**
 * Count working days (excluding weekends) between two dates (inclusive)
 */
export function countWorkingDays(startDate: Date, endDate: Date): number {
  const start = new Date(startDate);
  const end = new Date(endDate);
  
  if (start > end) {
    return 0;
  }
  
  let count = 0;
  const current = new Date(start);
  
  while (current <= end) {
    if (isWeekday(current)) {
      count++;
    }
    current.setDate(current.getDate() + 1);
  }
  
  return count;
}

/**
 * Get all weekdays between two dates (inclusive)
 */
export function getWorkingDaysBetween(startDate: Date, endDate: Date): Date[] {
  const start = new Date(startDate);
  const end = new Date(endDate);
  
  if (start > end) {
    return [];
  }
  
  const workingDays: Date[] = [];
  const current = new Date(start);
  
  while (current <= end) {
    if (isWeekday(current)) {
      workingDays.push(new Date(current));
    }
    current.setDate(current.getDate() + 1);
  }
  
  return workingDays;
}

/**
 * Format a date range for display
 */
export function formatDateRange(startDate: Date, endDate: Date): string {
  return `${startDate.toLocaleDateString()} - ${endDate.toLocaleDateString()}`;
}

/**
 * Check if a date is in range between start and end (inclusive)
 */
export function isDateInRange(date: Date, startDate: Date, endDate: Date): boolean {
  const d = new Date(date);
  const s = new Date(startDate);
  const e = new Date(endDate);
  
  // Normalize to midnight
  d.setHours(0, 0, 0, 0);
  s.setHours(0, 0, 0, 0);
  e.setHours(0, 0, 0, 0);
  
  return d >= s && d <= e;
}

/**
 * Check if two dates are the same day
 */
export function isSameDay(date1: Date, date2: Date): boolean {
  const d1 = new Date(date1);
  const d2 = new Date(date2);
  d1.setHours(0, 0, 0, 0);
  d2.setHours(0, 0, 0, 0);
  return d1.getTime() === d2.getTime();
}
