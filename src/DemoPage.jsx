'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAuth } from './App';
import { useToast } from '@/components/ui/use-toast';
import { Upload, Scissors, Search, Activity, Book, Combine, Loader2 } from 'lucide-react';

const ENDPOINTS = {
  instrument_detection: {
    path: 'detect_image_instruments',
    fileKey: 'image',
    accept: 'image/jpeg,image/png,image/jpg,image/webp',
    inputName: 'Image',
  },
  instrument_segmentation: {
    path: 'segment_image',
    fileKey: 'image',
    accept: 'image/jpeg,image/png,image/jpg,image/webp',
    inputName: 'Image',
  },
  atomic_actions: {
    path: 'atomic_actions',
    fileKey: 'video',
    accept: 'video/mp4,video/avi,video/mov',
    inputName: 'Video',
  },
  phase_step: {
    path: 'phase_step',
    fileKey: 'video',
    accept: 'video/mp4,video/avi,video/mov',
    inputName: 'Video',
  },
  combined_analysis: {
    path: 'analyze_surgical_video',
    fileKey: 'video',
    accept: 'video/mp4,video/mov,video/avi',
    inputName: 'Video',
  },
};

const FrameAnalysisDisplay = ({ frame }) => {
  const instruments = frame.detected_instruments?.join(', ') || 'None Detected';
  const detectionProbabilities = Object.entries(frame.detection_probabilities || {}).map(([key, prob]) => (
    <span key={key} className="inline-block bg-medical-light/50 text-xs px-2 py-0.5 rounded-full mr-1 mb-1">
      {key}: {(prob * 100).toFixed(1)}%
    </span>
  ));

  return (
    <div className="flex flex-col space-y-2 p-3 bg-white rounded-lg shadow-md border border-red-100">
      <div className="flex justify-between items-center border-b pb-1">
        <p className="text-sm font-semibold text-red-600">
          Frame @ {frame.timestamp_sec}
        </p>
        <p className="text-xs text-gray-500">
          Index: {frame.frame_index}
        </p>
      </div>

      <div className="relative w-full aspect-video rounded-md overflow-hidden border border-gray-200">
        <img
          src={`http://localhost:5000/${frame.segmented_image_path}`}
          alt={`Segmented Frame ${frame.frame_index}`}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = "https://placehold.co/512x288/CCCCCC/333333?text=Segmentation+Error";
          }}
        />
      </div>

      <div>
        <p className="text-sm font-medium text-medical-text mb-1">Instruments:</p>
        <div className="flex flex-wrap">
          {detectionProbabilities}
        </div>
      </div>
    </div>
  );
};

const DemoPage = ({ toggleSidebar, isSidebarOpen = false }) => {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);

  const { user } = useAuth();
  const { toast } = useToast() || { toast: (args) => console.log('Fallback Toast:', args) };

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) {
      setFile(null);
      setFileType(null);
      return;
    }

    if (uploadedFile.type.startsWith('video/')) setFileType('video');
    else if (uploadedFile.type.startsWith('image/')) setFileType('image');
    else {
      toast({
        variant: 'destructive',
        title: 'File Error',
        description: 'Unsupported file type uploaded.',
      });
      setFile(null);
      setFileType(null);
      return;
    }

    setFile(uploadedFile);
    setResult(null);
  };

  const handlePrediction = async (modelKey) => {
    setSelectedModel(modelKey);
    const config = ENDPOINTS[modelKey];

    if (fileType !== config.fileKey) {
      toast({
        variant: 'destructive',
        title: 'Input Error',
        description: `Model "${modelKey.replace('_', ' ')}" requires a ${config.fileKey}, but a ${fileType} was uploaded.`,
      });
      return;
    }

    if (!user) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: 'Please log in to use this feature.',
      });
      return;
    }

    if (!file) {
      toast({
        variant: 'destructive',
        title: 'Error',
        description: `Please upload a ${config.inputName}.`,
      });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append(config.fileKey, file);
    const token = localStorage.getItem('token');

    try {
      const endpoint = modelKey === 'combined_analysis'
        ? `http://localhost:5000/combined_inference/${config.path}`
        : `http://localhost:5000/predict/${config.path}`;

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          Origin: 'http://localhost:3000',
        },
        body: formData,
        credentials: 'include',
      });

      const data = await res.json();

      if (res.ok) {
        setResult({ ...data, model: modelKey });
        toast({
          title: 'Success',
          description: 'Analysis complete! Results saved to history.',
        });
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: data.error || 'Analysis failed',
        });
      }
    } catch (err) {
      console.error('[ERROR] Fetch error:', err);
      toast({
        variant: 'destructive',
        title: 'Error',
        description: `Error during analysis: ${err.message}`,
      });
    } finally {
      setLoading(false);
    }
  };

  const getResultSourcePath = () => {
    if (result?.model === 'instrument_detection' && result.input_path)
      return result.input_path;
    if (result?.model === 'instrument_segmentation' && result.output_path)
      return result.output_path;
    if ((result?.model === 'atomic_actions' || result?.model === 'phase_step') && result.input_path)
      return result.input_path;
    if (result?.model === 'combined_analysis' && result.output_path)
      return result.output_path;
    return null;
  };

  const resultSourcePath = getResultSourcePath();
  const uploadedPreview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="min-h-screen pt-20 pb-12 bg-medical-light/20">
      {user && !isSidebarOpen && (
        <motion.div
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
            className="fixed left-4 top-20 z-30 bg-medical hover:bg-medical/90 text-white shadow-lg"
          >
            <Book className="h-4 w-4 sm:h-5 sm:w-5 mr-1 sm:mr-2" />
            <span className="text-sm sm:text-base">History</span>
          </Button>
        </motion.div>
      )}

      <div
        className={`container mx-auto px-4 sm:px-6 lg:px-8 transition-all duration-300 ${
          isSidebarOpen ? 'ml-80 sm:ml-96' : ''
        }`}
      >
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ y: -30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-8"
          >
            <h1 className="pt-8 text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              Surgical AI Analysis Demo
            </h1>
            <p className="text-base sm:text-lg text-medical-text-light max-w-2xl mx-auto">
              Upload an image or video to run AI-based detection, segmentation, or action analysis.
            </p>
          </motion.div>

          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <Card className="medical-card border-0 shadow-lg">
              <CardHeader>
                <CardTitle className="flex items-center text-medical-text">
                  <Upload className="h-6 w-6 mr-2 text-medical" />
                  Upload Surgical Media
                </CardTitle>
              </CardHeader>

              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="file" className="text-medical-text font-medium">
                    Select Image or Video
                  </Label>
                  <Input
                    id="file"
                    type="file"
                    accept={`${ENDPOINTS.instrument_detection.accept},${ENDPOINTS.atomic_actions.accept}`}
                    onChange={handleFileChange}
                    className="border-medical-light focus:border-medical"
                  />
                </div>

                {/* First 4 Buttons in One Row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                  {[
                    { key: 'instrument_detection', icon: Search, label: 'Detect Instruments (Image)', short: 'Detect (Image)' },
                    { key: 'instrument_segmentation', icon: Scissors, label: 'Segment Instruments (Image)', short: 'Segment (Image)' },
                    { key: 'atomic_actions', icon: Activity, label: 'Detect Actions (Video)', short: 'Actions (Video)' },
                    { key: 'phase_step', icon: Activity, label: 'Phase-Step Analysis', short: 'Phase-Step' },
                  ].map((btn, idx) => {
                    const config = ENDPOINTS[btn.key];
                    const disabled = loading || !file || fileType !== config.fileKey;

                    return (
                      <motion.div
                        key={btn.key}
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ duration: 0.3, delay: idx * 0.08 }}
                        whileHover={{ scale: disabled ? 1 : 1.03 }}
                        whileTap={{ scale: disabled ? 1 : 0.97 }}
                        className="w-full"
                      >
                        <Button
                          onClick={() => handlePrediction(btn.key)}
                          disabled={disabled}
                          className={`
                            w-full h-12 px-4 py-3 text-sm font-medium rounded-lg shadow-md transition-all
                            bg-medical hover:bg-medical/90 text-white
                            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
                            flex items-center justify-center gap-2
                          `}
                        >
                          {loading && selectedModel === btn.key ? (
                            <Loader2 className="animate-spin h-5 w-5" />
                          ) : (
                            <btn.icon className="h-5 w-5" />
                          )}
                          <span className="hidden sm:inline">{btn.label}</span>
                          <span className="sm:hidden">{btn.short}</span>
                        </Button>
                      </motion.div>
                    );
                  })}
                </div>

                {/* Special Combined Analysis Button - Full Width, Below */}
                <div className="mt-4">
                  <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.3, delay: 0.35 }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="w-full"
                  >
                    <Button
                      onClick={() => handlePrediction('combined_analysis')}
                      disabled={loading || !file || fileType !== 'video'}
                      className={`
                        w-full h-14 px-6 py-3 text-base font-semibold rounded-xl shadow-lg transition-all
                        bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300
                        text-gray-800 border border-gray-300
                        ${loading || !file || fileType !== 'video' ? 'opacity-50 cursor-not-allowed' : ''}
                        flex items-center justify-center gap-3
                      `}
                    >
                      {loading && selectedModel === 'combined_analysis' ? (
                        <Loader2 className="animate-spin h-6 w-6" />
                      ) : (
                        <Combine className="h-6 w-6" />
                      )}
                      <span>Combined Analysis</span>
                    </Button>
                  </motion.div>
                </div>

                <AnimatePresence>
                  {loading && (
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="text-medical-text text-center flex items-center justify-center"
                    >
                      <Loader2 className="animate-spin h-5 w-5 mr-2" />
                      Analyzing... Please wait.
                    </motion.p>
                  )}
                </AnimatePresence>

                <AnimatePresence>
                  {result && resultSourcePath && (
                    <motion.div
                      key={result.model}
                      initial={{ y: 30, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      exit={{ y: -30, opacity: 0 }}
                      transition={{ duration: 0.4 }}
                      className="mt-6"
                    >
                      <h3 className="text-lg font-semibold text-medical-text mb-4">
                        Results for {result.model.replace('_', ' ').toUpperCase()}
                      </h3>

                      {fileType === 'image' ? (
                        result.model === 'instrument_detection' ? (
                          <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }} className="mt-3">
                            <p className="text-sm text-medical-text-light mb-1">Detected Image</p>
                            <img
                              src={`http://localhost:5000/${resultSourcePath}`}
                              alt="Detected"
                              className="w-full max-w-2xl mx-auto rounded-md border border-medical-light shadow-sm"
                            />
                          </motion.div>
                        ) : (
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-3">
                            {[
                              { title: 'Input Image', src: uploadedPreview },
                              { title: 'Segmented Output', src: `http://localhost:5000/${resultSourcePath}` },
                            ].map((img, i) => (
                              <motion.div
                                key={i}
                                initial={{ opacity: 0, x: i === 0 ? -30 : 30 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: i * 0.15 }}
                              >
                                <p className="text-sm text-medical-text-light mb-1">{img.title}</p>
                                <img
                                  src={img.src}
                                  alt={img.title}
                                  className="w-full h-auto rounded-md border border-medical-light shadow-sm"
                                />
                              </motion.div>
                            ))}
                          </div>
                        )
                      ) : (
                        (result.model === 'atomic_actions' || result.model === 'phase_step') && (
                          <motion.video
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            controls
                            className="w-full max-w-3xl mx-auto mt-4 rounded-md border border-medical-light shadow-sm"
                          >
                            <source src={`http://localhost:5000/${resultSourcePath}`} type="video/mp4" />
                          </motion.video>
                        )
                      )}

                      {result.model === 'instrument_detection' && result.instruments && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4 p-4 bg-gray-50 rounded-lg"
                        >
                          <p className="font-medium text-medical-text">Detected Instruments:</p>
                          <ul className="list-disc pl-5 mt-1 text-sm text-medical-text-light">
                            {result.instruments.map((inst, idx) => (
                              <li key={idx}>{inst}</li>
                            ))}
                          </ul>
                        </motion.div>
                      )}

                      {result.model === 'atomic_actions' && result.actions && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4 p-4 bg-gray-50 rounded-lg"
                        >
                          <p className="font-medium text-medical-text">Detected Actions:</p>
                          <ul className="list-disc pl-5 mt-1 text-sm text-medical-text-light">
                            {result.actions.map((action, idx) => (
                              <li key={idx}>{action}</li>
                            ))}
                          </ul>
                        </motion.div>
                      )}

                      {result.model === 'phase_step' && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-6 p-4 bg-white rounded-lg shadow border"
                        >
                          <h3 className="text-lg font-semibold text-medical-text">Phase-Step Results</h3>
                          <p><strong>Phase:</strong> {result.predicted_phase}</p>
                          <p className="text-sm text-gray-600">{result.phase_description}</p>
                          <p><strong>Step:</strong> {result.predicted_step}</p>
                          <p className="text-sm text-gray-600">{result.step_description}</p>
                        </motion.div>
                      )}

                      {result.model === 'combined_analysis' && result.analysis && (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.97 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="mt-8 p-6 bg-white rounded-lg shadow-xl border border-gray-200 max-w-5xl mx-auto"
                        >
                          <h3 className="text-2xl font-bold text-gray-800 text-center mb-6">
                            Complete Surgical AI Analysis
                          </h3>

                          <div className="grid md:grid-cols-2 gap-6 mb-8">
                            {result.input_path && (
                              <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
                                <p className="text-sm font-semibold text-medical-text mb-2">Input Video</p>
                                <video controls className="w-full rounded-lg border-2 border-medical-light shadow-md">
                                  <source src={`http://localhost:5000/${result.input_path}`} type="video/mp4" />
                                </video>
                              </motion.div>
                            )}
                            {result.output_path && (
                              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
                                <p className="text-sm font-semibold text-medical-text mb-2">Annotated Output</p>
                                <video controls className="w-full rounded-lg border-2 border-medical-light shadow-md">
                                  <source src={`http://localhost:5000/${result.output_path}`} type="video/mp4" />
                                </video>
                              </motion.div>
                            )}
                          </div>

                          <div className="grid md:grid-cols-2 gap-4 mb-6">
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="p-4 bg-gray-50 rounded-lg border"
                            >
                              <p className="text-lg font-semibold text-gray-700 mb-1">Current Phase</p>
                              <p className="text-xl font-bold text-medical">{result.analysis.phase.name}</p>
                              <p className="text-sm text-gray-600">{result.analysis.phase.description}</p>
                            </motion.div>
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.1 }}
                              className="p-4 bg-gray-50 rounded-lg border"
                            >
                              <p className="text-lg font-semibold text-gray-700 mb-1">Current Step</p>
                              <p className="text-xl font-bold text-medical">{result.analysis.step.name}</p>
                              <p className="text-sm text-gray-600">{result.analysis.step.description}</p>
                            </motion.div>
                          </div>

                          {result.analysis.actions?.length > 0 && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: 0.2 }}
                              className="text-center"
                            >
                              <p className="text-lg font-bold text-gray-700 mb-3">Detected Atomic Actions</p>
                              <div className="flex flex-wrap justify-center gap-2">
                                {result.analysis.actions.map((action, i) => (
                                  <motion.span
                                    key={i}
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ delay: i * 0.05 }}
                                    className="px-4 py-2 bg-medical text-white font-medium rounded-full text-sm"
                                  >
                                    {action}
                                  </motion.span>
                                ))}
                              </div>
                            </motion.div>
                          )}
                        </motion.div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;