import { useState } from 'react';
import {
      analyzeText,
      analyzeAudio,
      analyzeVision,
      analyzeFused,
      analyzeVideo,
      type TextResponse,
      type AudioResponse,
      type VisionResponse,
      type FusedResponse,
      type VideoResponse
} from '@/api/client';

export type AnalysisResult = {
      text?: TextResponse;
      audio?: AudioResponse;
      vision?: VisionResponse;
      fused?: FusedResponse;
      video?: VideoResponse;
};

export const useAnalysis = () => {
      const [loading, setLoading] = useState(false);
      const [error, setError] = useState<string | null>(null);
      const [results, setResults] = useState<AnalysisResult | null>(null);
      const [history, setHistory] = useState<AnalysisResult[]>([]);

      const analyze = async (
            text: string | null,
            audioFile: File | null,
            imageFile: File | null,
            videoFile: File | null
      ) => {
            setLoading(true);
            setError(null);
            setResults(null);

            try {
                  const newResults: AnalysisResult = {};

                  // Max Fusion (Video) - Exclusive or combined?
                  // If video is present, usually it supersedes others or is treated as Max Fusion
                  if (videoFile) {
                        newResults.video = await analyzeVideo(videoFile);
                        // Video analysis might return sub-modalities, we could populate them if we want
                        // But let's keep it under 'video' key for now as the API response structure handles it.
                  } else {
                        // Parallel execution for individual modalities
                        const promises = [];

                        if (text) {
                              promises.push(analyzeText(text).then((res: TextResponse) => { newResults.text = res; }));
                        }
                        if (audioFile) {
                              promises.push(analyzeAudio(audioFile).then((res: AudioResponse) => { newResults.audio = res; }));
                        }
                        if (imageFile) {
                              promises.push(analyzeVision(imageFile).then((res: VisionResponse) => { newResults.vision = res; }));
                        }

                        await Promise.all(promises);

                        // Fused Analysis (only if > 1 modality or specifically requested)
                        // Check if we have enough for fusion (at least one input)
                        if (text || audioFile || imageFile) {
                              try {
                                    // We pass the raw inputs to fused endpoint
                                    const fused = await analyzeFused(text || null, audioFile || null, imageFile || null);
                                    newResults.fused = fused;
                              } catch (e) {
                                    console.error("Fused analysis failed even though individual might have succeeded", e);
                                    // Don't fail the whole batch if fusion fails?
                              }
                        }
                  }

                  setResults(newResults);
                  setHistory(prev => [newResults, ...prev]);

            } catch (err: any) {
                  console.error(err);
                  setError(err.response?.data?.detail || err.message || "An error occurred during analysis");
            } finally {
                  setLoading(false);
            }
      };

      const clearResults = () => {
            setResults(null);
            setError(null);
      }

      return {
            analyze,
            loading,
            error,
            results,
            clearResults,
            history
      };
};
