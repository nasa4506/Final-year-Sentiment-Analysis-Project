import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Adjust if backend runs on different port

const api = axios.create({
      baseURL: API_BASE_URL,
      headers: {
            'Content-Type': 'application/json',
      },
});

export interface SentimentResponse {
      sentiment: string;
      confidence: number;
      details?: any;
      reasoning?: { word: string; weight: number }[];
      math_breakdown?: {
            modality: string;
            original_sentiment: string;
            confidence: number;
            weight: number;
            contribution: number;
            maps_to: string;
      }[];
}

export type TextResponse = SentimentResponse;
export type AudioResponse = SentimentResponse;
export type VisionResponse = SentimentResponse;
export type FusedResponse = SentimentResponse;

export interface VideoResponse {
      video_sentiment: string;
      video_confidence: number;
      modalities: {
            [key: string]: {
                  sentiment: string;
                  confidence: number;
                  details?: string;
                  reasoning?: { word: string; weight: number }[];
            };
      };
      frames_analyzed: number;
      transcription: string;
      math_breakdown?: {
            modality: string;
            original_sentiment: string;
            confidence: number;
            weight: number;
            contribution: number;
            maps_to: string;
      }[];
}

export const analyzeText = async (text: string): Promise<SentimentResponse> => {
      const response = await api.post<SentimentResponse>('/analyze/text', { text });
      return response.data;
};

export const analyzeAudio = async (file: File): Promise<SentimentResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await api.post<SentimentResponse>('/analyze/audio', formData, {
            headers: {
                  'Content-Type': 'multipart/form-data',
            },
      });
      return response.data;
};

export const analyzeVision = async (file: File): Promise<SentimentResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await api.post<SentimentResponse>('/analyze/vision', formData, {
            headers: {
                  'Content-Type': 'multipart/form-data',
            },
      });
      return response.data;
};

export const analyzeFused = async (
      text: string | null,
      audioFile: File | null,
      imageFile: File | null
): Promise<SentimentResponse> => {
      const formData = new FormData();
      if (text) formData.append('text', text);
      if (audioFile) formData.append('audio_file', audioFile);
      if (imageFile) formData.append('image_file', imageFile);

      const response = await api.post<SentimentResponse>('/analyze/fused', formData, {
            headers: {
                  'Content-Type': 'multipart/form-data',
            },
      });
      return response.data;
};

export const analyzeVideo = async (file: File): Promise<VideoResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await api.post<VideoResponse>('/analyze/video', formData, {
            headers: {
                  'Content-Type': 'multipart/form-data',
            },
      });
      return response.data;
};

export default api;
