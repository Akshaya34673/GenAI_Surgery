
// src/DemoPage.jsx
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAuth } from './App';
import { useToast } from '@/components/ui/use-toast'; // Ensure this path is correct
import { Upload, Scissors, Search, Activity, Book } from 'lucide-react';

const DemoPage = ({ toggleSidebar }) => {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const { user } = useAuth();
  const { toast } = useToast() || { toast: (args) => console.log('Fallback Toast:', args) }; // Enhanced fallback

  const handleVideoChange = (e) => {
    setVideo(e.target.files[0]);
    setResult(null);
  };

  const handlePrediction = async (endpoint, modelName) => {
    if (!user) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please log in to use this feature.' });
      return;
    }
    if (!video) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please upload a video.' });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('video', video);
    const token = localStorage.getItem('token');

    try {
      const res = await fetch(`http://localhost:5000/predict/${endpoint}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Origin': 'http://localhost:3000'
        },
        body: formData,
        credentials: 'include'
      });
      const data = await res.json();
      console.log('[DEBUG] Response from server:', data); // Debug log
      if (res.ok) {
        setResult({ ...data, model: modelName });
        toast({ title: 'Success', description: 'Analysis complete! Results saved to history.' });
      } else {
        toast({ variant: 'destructive', title: 'Error', description: data.error || 'Analysis failed' });
      }
    } catch (err) {
      console.error('[ERROR] Fetch error:', err);
      toast({ variant: 'destructive', title: 'Error', description: `Error during analysis: ${err.message}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-20 pb-12 bg-medical-light/20 flex">
      {user && (
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="fixed left-4 top-20 z-50 bg-medical hover:bg-medical/90 text-white"
        >
          <Book className="h-5 w-5 mr-2" /> History
        </Button>
      )}

      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-medical-text mb-4">
              Video Analysis Demo
            </h1>
            <p className="text-lg text-medical-text-light max-w-2xl mx-auto">
              Upload a surgical video clip to see our AI in action. Experience real-time instrument segmentation and action detection.
            </p>
          </div>

          <Card className="medical-card border-0">
            <CardHeader>
              <CardTitle className="flex items-center text-medical-text">
                <Upload className="h-6 w-6 mr-2 text-medical" />
                Upload Surgical Video
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="video" className="text-medical-text font-medium">Select Video</Label>
                <Input 
                  id="video" 
                  type="file" 
                  accept="video/*" 
                  onChange={handleVideoChange}
                  className="border-medical-light focus:border-medical"
                />
              </div>

             
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Button 
                  onClick={() => handlePrediction('detect_instruments', 'instrument_detection')}  // Updated endpoint
                  disabled={loading || !video}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center"
                >
                  <Search className="h-5 w-5 mr-2" /> Detect Instruments
                </Button>
                <Button 
                  onClick={() => handlePrediction('instrument_segmentation', 'instrument_segmentation')}
                  disabled={loading || !video}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center"
                >
                  <Scissors className="h-5 w-5 mr-2" /> Segment Instruments
                </Button>
                <Button 
                  onClick={() => handlePrediction('atomic_actions', 'atomic_actions')}
                  disabled={loading || !video}
                  className="bg-medical hover:bg-medical/90 medical-button flex items-center"
                >
                  <Activity className="h-5 w-5 mr-2" /> Detect Atomic Actions
                </Button>
              </div>

              {loading && <p className="text-medical-text">Analyzing...</p>}

              {result && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-medical-text">Results for {result.model.replace('_', ' ').toUpperCase()}</h3>
                  <p>Video: {result.video_path.split('/').pop()}</p>
                  {result.model === 'instrument_detection' && (
                    <div>
                      <p>Detected Instruments:</p>
                      <ul className="list-disc pl-5 mt-2">
                        {result.instruments && result.instruments.length > 0 ? (
                          result.instruments.map((inst, idx) => (
                            <li key={idx}>{inst}</li>
                          ))
                        ) : (
                          <li>No instruments detected.</li>
                        )}
                      </ul>
                    </div>
                  )}
                  {result.model === 'instrument_segmentation' && result.segmented_frames && (
                    <div>
                      <p>Segmented Frames:</p>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        {result.segmented_frames.map((frame, idx) => (
                          <img key={idx} src={`http://localhost:5000/${frame}`} alt={`Frame ${idx}`} className="w-full h-auto" />
                        ))}
                      </div>
                    </div>
                  )}
                  {result.model === 'atomic_actions' && result.actions && (
                    <div>
                      <p>Detected Atomic Actions:</p>
                      <ul className="list-disc pl-5 mt-2">
                        {result.actions.map((action, idx) => (
                          <li key={idx}>{action}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default DemoPage;
