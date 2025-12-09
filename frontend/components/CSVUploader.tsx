import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Upload, AlertCircle, CheckCircle, FileIcon } from 'lucide-react';
import { toast } from 'sonner';
import { API_ENDPOINTS, fetchAPI } from '../lib/api';

interface CSVUploaderProps {
  onSuccess?: () => void;
}

export function CSVUploader({ onSuccess }: CSVUploaderProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    
    if (!file) return;

    // Validazione: solo CSV
    if (!file.name.endsWith('.csv')) {
      toast.error('Please select a CSV file');
      return;
    }

    // Validazione: massimo 10MB
    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    setUploadedFile(file);
    setUploadProgress(0);
  };

  const handleUpload = async () => {
    if (!uploadedFile) {
      toast.error('Please select a file first');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      // Simulare il progresso dell'upload
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 30;
        });
      }, 300);

      // Costruire l'URL completo dell'API
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const uploadUrl = `${apiUrl}/api/csv/upload`;

      const response = await fetch(uploadUrl, {
        method: 'POST',
        body: formData,
        // NON impostare Content-Type: il browser lo farà automaticamente con boundary
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const error = await response.json().catch(() => ({ message: 'Upload failed' }));
        throw new Error(error.message || error.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      
      toast.success(`CSV uploaded successfully! ${data.rows_processed || ''} rows processed.`);
      
      // Reset del form
      setUploadedFile(null);
      setUploadProgress(0);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      // Callback opzionale
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(
        error instanceof Error 
          ? error.message 
          : 'Failed to upload CSV. Please check the file format.'
      );
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleClear = () => {
    setUploadedFile(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Card className="border-dashed">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Bulk Upload CSV Data
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* File Input Area */}
        <div>
          <Label htmlFor="csv-file" className="block mb-4">
            Select CSV File
          </Label>
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              id="csv-file"
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              disabled={isUploading}
              className="hidden"
            />
            {uploadedFile ? (
              <div className="flex flex-col items-center gap-2">
                <FileIcon className="w-8 h-8 text-blue-500" />
                <p className="text-sm font-medium text-gray-900">{uploadedFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(uploadedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            ) : (
              <div>
                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-gray-600">
                  Drag and drop your CSV file here, or click to select
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Maximum file size: 10 MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Uploading...</span>
              <span>{Math.round(uploadProgress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Success Message */}
        {uploadProgress === 100 && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex gap-3">
            <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-green-800">
              File uploaded successfully!
            </div>
          </div>
        )}

        {/* Expected CSV Format */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
            <span className="text-xs font-semibold text-blue-900">Expected CSV Format</span>
          </div>
          <div className="text-xs text-blue-800 space-y-2">
            <div>Your CSV should have these columns:</div>
            <code className="block bg-white p-2 rounded mt-1 font-mono text-xs overflow-x-auto">
              school, date, dish_name, portions_prepared, portions_wasted
            </code>
            <div className="mt-2">Example:</div>
            <code className="block bg-white p-2 rounded mt-1 font-mono text-xs overflow-x-auto">
              Lincoln Elementary,2024-11-10,Spaghetti,120,15
            </code>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 pt-4">
          <Button
            onClick={handleUpload}
            disabled={!uploadedFile || isUploading}
            className="flex-1"
          >
            {isUploading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Upload CSV
              </>
            )}
          </Button>
          {uploadedFile && (
            <Button
              onClick={handleClear}
              variant="outline"
              disabled={isUploading}
            >
              Clear
            </Button>
          )}
        </div>

        {/* Info */}
        <div className="text-xs text-gray-600 text-center">
          <p>Files are processed in the background. You'll receive a notification when complete.</p>
        </div>
      </CardContent>
    </Card>
  );
}
