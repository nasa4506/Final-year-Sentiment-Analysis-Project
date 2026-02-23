import { MainSurface } from "@/components/layout/MainSurface"
import { AnalysisInput } from "@/features/AnalysisInput"
import { EvidencePanel } from "@/features/EvidencePanel"
import { useAnalysis } from "@/hooks/useAnalysis"

function App() {
  const { analyze, loading, results, error } = useAnalysis()

  return (
    <MainSurface>
      <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full pt-10 pb-20">
        <div className="text-center space-y-2 mb-4">
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-50 to-gray-400">
            Multimodal Sentiment Analysis
          </h1>
          <p className="text-gray-400">
            Drop text, audio, images, or video to detect underlying emotions.
          </p>
        </div>

        <AnalysisInput onAnalyze={analyze} loading={loading} />

        {error && (
          <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-200 text-sm text-center">
            {error}
          </div>
        )}

        <EvidencePanel results={results} loading={loading} />
      </div>
    </MainSurface>
  )
}

export default App
