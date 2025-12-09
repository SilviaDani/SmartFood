import React, { useState, useMemo } from 'react';
import { Button } from './ui/button';
import { cn } from './ui/utils';
import { DayPicker } from 'react-day-picker';
import { it } from 'date-fns/locale';
import { format } from 'date-fns';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { buttonVariants } from './ui/button';
import {
  isWeekend,
  isDateInRange,
  isSameDay,
  countWorkingDays,
} from '../lib/dateUtils';

interface DateRangeCalendarProps {
  startDate: Date | null;
  endDate: Date | null;
  onStartDateChange: (date: Date) => void;
  onEndDateChange: (date: Date) => void;
  onConfirm: () => void;
}

export function DateRangeCalendar({
  startDate,
  endDate,
  onStartDateChange,
  onEndDateChange,
  onConfirm,
}: DateRangeCalendarProps) {
  const [selectionPhase, setSelectionPhase] = useState<'start' | 'end'>('start');

  const handleDateClick = (date: Date | undefined) => {
    if (!date) return;

    // Non permettere la selezione di weekend
    if (isWeekend(date)) {
      return;
    }

    // Non permettere date passate
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    if (date < today) {
      return;
    }

    if (selectionPhase === 'start') {
      onStartDateChange(date);
      setSelectionPhase('end');
    } else {
      // Assicurati che endDate sia >= startDate
      if (startDate && date < startDate) {
        // Se l'utente seleziona una data prima della start, la nuova data diventa la start
        onStartDateChange(date);
        onEndDateChange(startDate);
      } else {
        onEndDateChange(date);
      }
    }
  };

  const workingDaysCount = useMemo(
    () => (startDate && endDate ? countWorkingDays(startDate, endDate) : 0),
    [startDate, endDate]
  );

  return (
    <div className="space-y-2 p-2 w-fit">
      {/* Display current selection */}
      <div className="bg-blue-50 border border-blue-200 rounded p-2 text-sm">
        <div className="font-semibold text-blue-900 mb-2">
          {selectionPhase === 'start'
            ? 'Seleziona data d\'inizio'
            : 'Seleziona data di fine'}
        </div>

        <div className="space-y-1 text-xs">
          {startDate && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0"></div>
              <span className="text-blue-800">
                <span className="font-semibold">{format(startDate, 'PPP', { locale: it })}</span>
              </span>
            </div>
          )}
          {endDate && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0"></div>
              <span className="text-blue-800">
                <span className="font-semibold">{format(endDate, 'PPP', { locale: it })}</span>
              </span>
            </div>
          )}
          {startDate && endDate && (
            <div className="flex items-center gap-2 pt-1 border-t border-blue-200">
              <span className="text-blue-800">
                Giorni lavorativi selezionati: <span className="font-bold text-blue-900">{workingDaysCount}</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Calendar - using the same styling as the library Calendar component */}
      <div className="flex justify-center">
        <DayPicker
          mode="single"
          selected={selectionPhase === 'start' ? startDate || undefined : endDate || undefined}
          onDayClick={handleDateClick}
          disabled={(date) => {
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            return isWeekend(date) || date < today;
          }}
          locale={it}
          showOutsideDays={true}
          defaultMonth={startDate || endDate || new Date()}
          className={cn("p-3")}
          classNames={{
            months: "flex flex-col sm:flex-row gap-2",
            month: "flex flex-col gap-4",
            caption: "flex justify-center pt-1 relative items-center w-full",
            caption_label: "text-sm font-medium",
            nav: "flex items-center gap-1",
            nav_button: cn(
              buttonVariants({ variant: "outline" }),
              "size-7 bg-transparent p-0 opacity-50 hover:opacity-100",
            ),
            nav_button_previous: "absolute left-1",
            nav_button_next: "absolute right-1",
            table: "w-full border-collapse space-x-1",
            head_row: "flex",
            head_cell:
              "text-muted-foreground rounded-md w-8 font-normal text-[0.8rem]",
            row: "flex w-full mt-2",
            cell: cn(
              "relative p-0 text-center text-sm focus-within:relative focus-within:z-20"
            ),
            day: cn(
              buttonVariants({ variant: "ghost" }),
              "size-8 p-0 font-normal aria-selected:opacity-100",
            ),
            day_selected:
              "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
            day_today: "bg-accent text-accent-foreground",
            day_outside:
              "day-outside text-muted-foreground aria-selected:text-muted-foreground",
            day_disabled: "text-muted-foreground opacity-50",
            day_hidden: "invisible",
          }}
          modifiers={{
            rangeStart: (date: Date) => startDate && isSameDay(date, startDate),
            rangeEnd: (date: Date) => endDate && isSameDay(date, endDate),
            rangeMid: (date: Date) =>
              startDate && endDate && isDateInRange(date, startDate, endDate),
          }}
          modifiersStyles={{
            rangeStart: {
              backgroundColor: '#3b82f6',
              color: 'white',
            },
            rangeEnd: {
              backgroundColor: '#10b981',
              color: 'white',
            },
            rangeMid: {
              backgroundColor: '#dbeafe',
              color: '#1e40af',
            },
          }}
          components={{
            IconLeft: ({ className, ...props }) => (
              <ChevronLeft className={cn("size-4", className)} {...props} />
            ),
            IconRight: ({ className, ...props }) => (
              <ChevronRight className={cn("size-4", className)} {...props} />
            ),
          }}
        />
      </div>

      {/* Info about weekends */}
      <div className="bg-red-50 border border-red-200 rounded p-2 flex gap-2 text-xs">
        <span className="text-red-600 leading-none flex-shrink-0">⚠️</span>
        <p className="text-red-800">
          Weekend (sab-dom) non selezionabili ed esclusi dalle previsioni.
        </p>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2 pt-1">
        <Button
          onClick={onConfirm}
          disabled={!startDate || !endDate}
          className="flex-1 h-8 text-xs"
        >
          Conferma
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            onStartDateChange(null as any);
            onEndDateChange(null as any);
            setSelectionPhase('start');
          }}
          className="flex-1 h-8 text-xs"
        >
          Reset
        </Button>
      </div>
    </div>
  );
}
