
// HomePage.jsx
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Link } from 'react-router-dom';
import { 
  Brain, 
  Zap, 
  Target, 
  Shield, 
  GraduationCap, 
  Workflow, 
  BarChart3,
  Phone,
  Mail,
  MessageSquare,
  Star,
  Send,
  Loader2
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import heroImage from '@/assets/hero-medical-ai.jpg';

const Home = () => {
  const [feedback, setFeedback] = useState({
    name: '',
    email: '',
    phone: '',
    message: '',
    rating: 5,
    category: 'contact'
  });
  const [loading, setLoading] = useState(false);
  const { toast: toastFn } = useToast() || { toast: (args) => {
    console.error('Toast function unavailable:', args);
    return null;
  }};

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFeedback(prev => ({ ...prev, [name]: value }));
  };

  const handleRatingChange = (rating) => {
    setFeedback(prev => ({ ...prev, rating }));
  };

  const handleSubmitFeedback = async (e) => {
    e.preventDefault();
    
    if (!feedback.name.trim() || !feedback.email.trim() || !feedback.message.trim()) {
      toastFn({
        variant: "destructive",
        title: "Missing Fields",
        description: "Please fill in all required fields.",
      });
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/auth/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback),
      });

      const data = await response.json();

      if (response.ok) {
        toastFn({
          title: "Success!",
          description: data.message || "Thank you for your message! We'll get back to you soon.",
          className: "border-green-500 bg-green-50 text-green-700",
        });
        setFeedback({
          name: '',
          email: '',
          phone: '',
          message: '',
          rating: 5,
          category: 'contact'
        });
      } else {
        toastFn({
          variant: "destructive",
          title: "Submission Failed",
          description: data.error || "Please try again later.",
        });
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
      toastFn({
        variant: "destructive",
        title: "Network Error",
        description: "Unable to connect to server. Please check your connection and try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-24 pb-20 overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${heroImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-medical/90 to-medical/70"></div>
        </div>
        
        <div className="relative container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
              Precision Surgical Insights with{' '}
              <span className="text-medical-light">GenAI</span>
            </h1>
            <p className="text-xl sm:text-2xl text-white/90 mb-8 leading-relaxed">
              Leveraging advanced GenAI for real-time scene understanding, tool, and organ detection in medical images.
            </p>
            <Button 
              asChild 
              size="lg" 
              className="hero-button"
            >
              <Link to="/demo">Try the Demo</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* How Our AI Works */}
      <section className="py-20 bg-medical-light/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              How Our AI Works
            </h2>
            <p className="text-lg text-medical-text-light max-w-2xl mx-auto">
              Advanced machine learning algorithms trained on thousands of surgical procedures
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="medical-card transition-all duration-300 border-0">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-medical/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Brain className="h-8 w-8 text-medical" />
                </div>
                <h3 className="text-xl font-semibold text-medical-text mb-4">AI Model Training</h3>
                <p className="text-medical-text-light leading-relaxed">
                  Our models are trained on diverse surgical datasets, ensuring accurate recognition across various procedures and anatomical structures.
                </p>
              </CardContent>
            </Card>

            <Card className="medical-card transition-all duration-300 border-0">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-medical/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Zap className="h-8 w-8 text-medical" />
                </div>
                <h3 className="text-xl font-semibold text-medical-text mb-4">Realtime Processing</h3>
                <p className="text-medical-text-light leading-relaxed">
                  Process surgical footage in real-time, providing instant feedback on instrument detection and procedural analysis.
                </p>
              </CardContent>
            </Card>

            <Card className="medical-card transition-all duration-300 border-0">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-medical/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Target className="h-8 w-8 text-medical" />
                </div>
                <h3 className="text-xl font-semibold text-medical-text mb-4">Enhanced Accuracy</h3>
                <p className="text-medical-text-light leading-relaxed">
                  Achieve over 95% accuracy in tool recognition and anatomical identification, continuously improving with each analysis.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
              Benefits for Surgical Practice
            </h2>
            <p className="text-lg text-medical-text-light max-w-2xl mx-auto">
              Transform your surgical workflows with AI-powered insights
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-medical-success/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <Shield className="h-6 w-6 text-medical-success" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-medical-text mb-2">Improved Patient Safety</h3>
                  <p className="text-medical-text-light">
                    Real-time monitoring and alerts help prevent complications and ensure optimal patient outcomes during surgery.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-medical-accent/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <GraduationCap className="h-6 w-6 text-medical-accent" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-medical-text mb-2">Enhanced Training & Simulation</h3>
                  <p className="text-medical-text-light">
                    Provide detailed feedback and analysis for medical education and surgical skill development.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-medical/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <Workflow className="h-6 w-6 text-medical" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-medical-text mb-2">Streamlined Workflows</h3>
                  <p className="text-medical-text-light">
                    Automate documentation and analysis, reducing administrative burden and improving efficiency.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-medical-accent/10 rounded-full flex items-center justify-center flex-shrink-0">
                  <BarChart3 className="h-6 w-6 text-medical-accent" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-medical-text mb-2">Post-operative Analysis</h3>
                  <p className="text-medical-text-light">
                    Comprehensive surgical review and performance analytics for continuous improvement.
                  </p>
                </div>
              </div>
            </div>

            <Card className="medical-card p-8 border-0">
              <CardContent className="p-0">
                <h3 className="text-2xl font-bold text-medical-text mb-6 text-center">
                  Surgical Workflow Enhancement
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-medical-light/50 rounded-lg">
                    <span className="text-medical-text font-medium">Pre-operative Planning</span>
                    <div className="w-3 h-3 bg-medical-success rounded-full"></div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-medical-light/50 rounded-lg">
                    <span className="text-medical-text font-medium">Real-time Analysis</span>
                    <div className="w-3 h-3 bg-medical-accent rounded-full"></div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-medical-light/50 rounded-lg">
                    <span className="text-medical-text font-medium">Post-operative Review</span>
                    <div className="w-3 h-3 bg-medical rounded-full"></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Get in Touch Section */}
      <section className="py-20 bg-medical-light/20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-3xl sm:text-4xl font-bold text-medical-text mb-4">
                Get in Touch
              </h2>
              <p className="text-lg text-medical-text-light">
                We'd love to hear from you! Share your feedback, ask questions, or get in touch with our team.
              </p>
            </div>

            <div className="mb-6 p-4 bg-medical/5 rounded-lg border border-medical-light/30">
              <Label className="text-sm font-medium text-medical-text mb-2 block">
                How would you rate your experience so far? (Optional)
              </Label>
              <div className="flex justify-center space-x-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <Star
                    key={star}
                    className={`h-6 w-6 cursor-pointer transition-all duration-200 ${
                      feedback.rating >= star
                        ? 'text-yellow-400 fill-yellow-400'
                        : 'text-gray-300 hover:text-yellow-400'
                    }`}
                    onClick={() => handleRatingChange(star)}
                    fill={feedback.rating >= star ? 'currentColor' : 'none'}
                  />
                ))}
              </div>
            </div>

            <Card className="medical-card border-0 shadow-lg">
              <CardContent className="p-8">
                <form onSubmit={handleSubmitFeedback} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Label htmlFor="name" className="text-medical-text font-medium">
                        Full Name *
                      </Label>
                      <Input 
                        id="name"
                        name="name"
                        type="text"
                        value={feedback.name}
                        onChange={handleInputChange}
                        placeholder="Enter your full name"
                        required
                        disabled={loading}
                        className="border-medical-light focus:border-medical transition-colors"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="phone" className="text-medical-text font-medium">
                        Phone (Optional)
                      </Label>
                      <div className="relative">
                        <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-medical-text-light" />
                        <Input 
                          id="phone"
                          name="phone"
                          type="tel"
                          value={feedback.phone}
                          onChange={handleInputChange}
                          placeholder="+1 (555) 123-4567"
                          disabled={loading}
                          className="pl-10 border-medical-light focus:border-medical transition-colors"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="email" className="text-medical-text font-medium">
                      Email Address *
                    </Label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-medical-text-light" />
                      <Input 
                        id="email"
                        name="email"
                        type="email"
                        value={feedback.email}
                        onChange={handleInputChange}
                        placeholder="your.email@example.com"
                        required
                        disabled={loading}
                        className="pl-10 border-medical-light focus:border-medical transition-colors"
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="message" className="text-medical-text font-medium">
                      Message *
                    </Label>
                    <div className="relative">
                      <MessageSquare className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                      <Textarea 
                        id="message"
                        name="message"
                        value={feedback.message}
                        onChange={handleInputChange}
                        placeholder="Tell us about your needs, feedback, or questions. We're here to help!"
                        rows={4}
                        required
                        disabled={loading}
                        className="pl-10 border-medical-light focus:border-medical resize-none transition-colors"
                      />
                    </div>
                  </div>
                  
                  <Button 
                    type="submit" 
                    disabled={loading || !feedback.name.trim() || !feedback.email.trim() || !feedback.message.trim()}
                    className="w-full bg-medical hover:bg-medical/90 disabled:bg-gray-400 disabled:cursor-not-allowed medical-button flex items-center justify-center transition-all"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Sending Message...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4 mr-2" />
                        Send Message
                      </>
                    )}
                  </Button>
                </form>

                <p className="mt-6 pt-4 border-t border-medical-light/30 text-xs text-medical-text-light text-center">
                  * Required fields. Your information is secure and will only be used to respond to your inquiry. 
                  We respect your privacy and comply with GDPR regulations.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-medical-text text-white py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-white/80 mb-4">
              © 2025 GenAI Surgical AI. All rights reserved.
            </p>
            <div className="flex justify-center space-x-6">
              <Link to="/terms" className="text-white/60 hover:text-white transition-colors">
                Terms of Service
              </Link>
              <Link to="/privacy" className="text-white/60 hover:text-white transition-colors">
                Privacy Policy
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;
