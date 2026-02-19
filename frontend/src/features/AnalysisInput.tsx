import * as React from "react"
import { Card } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { InputToken } from "@/components/ui/InputToken"
import { Textarea } from "@/components/ui/Textarea" // Need to create this or use standard
import { Mic, Image as ImageIcon, Video, Send } from "lucide-react"
import { AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

// Simple Textarea component inline for now or extract later



interface AnalysisInputProps {
      onAnalyze: (text: string | null, audio: File | null, image: File | null, video: File | null) => void
      loading: boolean
}

export const AnalysisInput: React.FC<AnalysisInputProps> = ({ onAnalyze, loading }) => {
      const [text, setText] = React.useState("")
      const [audioFile, setAudioFile] = React.useState<File | null>(null)
      const [imageFile, setImageFile] = React.useState<File | null>(null)
      const [videoFile, setVideoFile] = React.useState<File | null>(null)

      const audioInputRef = React.useRef<HTMLInputElement>(null)
      const imageInputRef = React.useRef<HTMLInputElement>(null)
      const videoInputRef = React.useRef<HTMLInputElement>(null)

      const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'audio' | 'image' | 'video') => {
            if (e.target.files && e.target.files[0]) {
                  const file = e.target.files[0]
                  if (type === 'audio') setAudioFile(file)
                  if (type === 'image') setImageFile(file)
                  if (type === 'video') {
                        setVideoFile(file)
                        // Max Fusion usually overrides others, but we can keep them in state and let hook decide or clear them.
                        // Let's clear others to avoid confusion for the user, as Video implies full context.
                        setAudioFile(null)
                        setImageFile(null)
                        setText("")
                  }
            }
      }

      const handleRemove = (type: 'audio' | 'image' | 'video') => {
            if (type === 'audio') { setAudioFile(null); if (audioInputRef.current) audioInputRef.current.value = "" }
            if (type === 'image') { setImageFile(null); if (imageInputRef.current) imageInputRef.current.value = "" }
            if (type === 'video') { setVideoFile(null); if (videoInputRef.current) videoInputRef.current.value = "" }
      }

      const handleSubmit = () => {
            if (!text && !audioFile && !imageFile && !videoFile) return
            onAnalyze(
                  text || null,
                  audioFile,
                  imageFile,
                  videoFile
            )
      }

      return (
            <Card className="p-1 overflow-visible border-primary/20 bg-surface/80 backdrop-blur-md sticky top-4 z-50">
                  <div className="flex flex-col gap-2 p-3">
                        {/* Token Area */}
                        <div className="flex flex-wrap gap-2 min-h-[32px]">
                              <AnimatePresence>
                                    {videoFile ? (
                                          <InputToken
                                                key="video"
                                                icon={Video}
                                                label="Video"
                                                value={videoFile.name}
                                                isActive={true}
                                                color="bg-red-500/20 text-red-200"
                                                onRemove={() => handleRemove('video')}
                                                onClick={() => { }}
                                          />
                                    ) : (
                                          <>
                                                {audioFile && (
                                                      <InputToken
                                                            key="audio"
                                                            icon={Mic}
                                                            label="Audio"
                                                            value={audioFile.name}
                                                            isActive={true}
                                                            color="bg-purple-500/20 text-purple-200"
                                                            onRemove={() => handleRemove('audio')}
                                                            onClick={() => { }}
                                                      />
                                                )}
                                                {imageFile && (
                                                      <InputToken
                                                            key="image"
                                                            icon={ImageIcon}
                                                            label="Image"
                                                            value={imageFile.name}
                                                            isActive={true}
                                                            color="bg-orange-500/20 text-orange-200"
                                                            onRemove={() => handleRemove('image')}
                                                            onClick={() => { }}
                                                      />
                                                )}
                                          </>
                                    )}
                              </AnimatePresence>

                              {!videoFile && !audioFile && !imageFile && !text && (
                                    <span className="text-sm text-gray-500 py-1.5 px-2">
                                          Select a modality to begin...
                                    </span>
                              )}
                        </div>

                        {/* Text Area (Hidden if Video is active to enforce Max Fusion purity, or allowed? Let's hide for simplicity) */}
                        {!videoFile && (
                              <Textarea
                                    placeholder="Type your text analysis here..."
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                    className="text-lg bg-transparent border-none focus:ring-0 px-2 min-h-[60px]"
                              />
                        )}

                        {/* Bottom Actions */}
                        <div className="flex justify-between items-center pt-2 border-t border-white/5">
                              <div className="flex gap-1">
                                    {!videoFile && (
                                          <>
                                                <Button variant="ghost" size="icon" onClick={() => audioInputRef.current?.click()} title="Add Audio">
                                                      <Mic className="h-5 w-5 text-purple-400" />
                                                </Button>
                                                <Button variant="ghost" size="icon" onClick={() => imageInputRef.current?.click()} title="Add Image">
                                                      <ImageIcon className="h-5 w-5 text-orange-400" />
                                                </Button>
                                          </>
                                    )}
                                    <Button variant="ghost" size="icon" onClick={() => videoInputRef.current?.click()} title="Add Video (Max Fusion)">
                                          <Video className={cn("h-5 w-5", videoFile ? "text-red-400" : "text-gray-400 hover:text-red-400")} />
                                    </Button>
                              </div>

                              <Button
                                    onClick={handleSubmit}
                                    disabled={loading || (!text && !audioFile && !imageFile && !videoFile)}
                                    className={cn("gap-2 transition-all", loading ? "opacity-80" : "")}
                              >
                                    {loading ? "Analyzing..." : "Run Analysis"}
                                    {!loading && <Send className="h-4 w-4" />}
                              </Button>
                        </div>
                  </div>

                  {/* Hidden file inputs */}
                  <input type="file" ref={audioInputRef} className="hidden" accept="audio/*" onChange={(e) => handleFileChange(e, 'audio')} />
                  <input type="file" ref={imageInputRef} className="hidden" accept="image/*" onChange={(e) => handleFileChange(e, 'image')} />
                  <input type="file" ref={videoInputRef} className="hidden" accept="video/*" onChange={(e) => handleFileChange(e, 'video')} />
            </Card>
      )
}
