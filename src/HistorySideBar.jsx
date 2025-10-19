// HistorySidebar.jsx
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { X, Film, Trash2 } from 'lucide-react';

const HistorySidebar = ({ onClose }) => {
  const [history, setHistory] = useState([]);
  const toastContext = useToast() || { toast: (args) => console.log('Toast:', args) }; // Fallback
  const { toast } = toastContext;

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      toast({ variant: 'destructive', title: 'Error', description: 'Please log in to view history.' });
      return;
    }

    fetch('http://localhost:5000/auth/history', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Origin': 'http://localhost:3000'
      },
      credentials: 'include' // Ensure credentials are sent
    })
      .then(res => res.json())
      .then(data => {
        if (data.history) {
          setHistory(data.history);
        } else {
          toast({ variant: 'destructive', title: 'Error', description: data.error || 'Failed to load history' });
        }
      })
      .catch(err => {
        console.error('History fetch error:', err);
        toast({ variant: 'destructive', title: 'Error', description: 'Error loading history' });
      });
  }, [toast]);

  const handleDelete = async (historyId) => {
    const token = localStorage.getItem('token');
    try {
      const res = await fetch(`http://localhost:5000/auth/history/${historyId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Origin': 'http://localhost:3000'
        },
        credentials: 'include' // Ensure credentials are sent
      });
      const data = await res.json();
      if (res.ok) {
        setHistory(history.filter(entry => entry._id !== historyId));
        toast({ title: 'Success', description: 'History entry deleted.' });
      } else {
        toast({ variant: 'destructive', title: 'Error', description: data.error || 'Delete failed' });
      }
    } catch (err) {
      toast({ variant: 'destructive', title: 'Error', description: `Error deleting history: ${err.message}` });
    }
  };

  return (
    <div className="fixed top-20 left-0 h-[calc(100%-5rem)] w-80 bg-background shadow-lg p-4 z-50 overflow-y-auto">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-medical-text">Analysis History</h2>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-5 w-5" />
        </Button>
      </div>
      {history.length === 0 ? (
        <p className="text-medical-text-light">No history available.</p>
      ) : (
        history.map((entry, idx) => (
          <Card key={idx} className="mb-4">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <Film className="h-5 w-5 text-medical mr-2" />
                  <p className="font-medium text-medical-text">{entry.model.replace('_', ' ').toUpperCase()}</p>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => handleDelete(entry._id)}
                  className="text-red-500 hover:text-red-700"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <p className="text-sm text-medical-text-light">Video: {entry.video_path.split('/').pop()}</p>
              {entry.model === 'instrument_segmentation' ? (
                <div>
                  <p className="text-sm text-medical-text-light">Segmented Frames:</p>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {entry.result.map((frame, frameIdx) => (
                      <img key={frameIdx} src={`http://localhost:5000/${frame}`} alt={`Frame ${frameIdx}`} className="w-full h-auto" />
                    ))}
                  </div>
                </div>
              ) : entry.model === 'atomic_actions' ? (
                <div>
                  <p className="text-sm text-medical-text-light">Actions:</p>
                  <ul className="list-disc pl-5 text-sm text-medical-text-light">
                    {entry.result.map((action, idx) => (
                      <li key={idx}>{action}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-sm text-medical-text-light">Results: {entry.result.join(', ')}</p>
              )}
              <p className="text-sm text-medical-text-light">Time: {new Date(entry.timestamp).toLocaleString()}</p>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
};

export default HistorySidebar;