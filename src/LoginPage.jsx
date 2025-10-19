import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Mail, Lock, Phone, Stethoscope } from 'lucide-react';
import { useAuth } from './App';
import { useToast } from '@/components/ui/use-toast';

const LoginPage = () => {
  const [isOtpLogin, setIsOtpLogin] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [isOtpSent, setIsOtpSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const { setUser } = useAuth();
  const navigate = useNavigate();
  const toast = useToast(); // Updated: No destructuring, assuming it returns the Sonner toast function

  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': 'http://localhost:3000',
        },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (res.ok) {
        localStorage.setItem('token', data.token);
        setUser(data.user);
        toast.success("Logged in successfully!"); // Updated to Sonner syntax
        navigate('/');
      } else {
        toast.error(data.error || "Login failed"); // Updated to Sonner syntax
      }
    } catch (err) {
      console.error('Login error:', err);
      toast.error("An error occurred during login."); // Updated to Sonner syntax
    } finally {
      setLoading(false);
    }
  };

  const handleSendOtp = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/auth/send-otp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': 'http://localhost:3000',
        },
        body: JSON.stringify({ phone }),
      });
      if (res.ok) {
        setIsOtpSent(true);
        toast.success("OTP sent!"); // Updated to Sonner syntax
      } else {
        const data = await res.json();
        toast.error(data.error || "Failed to send OTP"); // Updated to Sonner syntax
      }
    } catch (err) {
      console.error('OTP send error:', err);
      toast.error("An error occurred while sending OTP."); // Updated to Sonner syntax
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOtp = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/auth/verify-otp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': 'http://localhost:3000',
        },
        body: JSON.stringify({ phone, code: otp }),
      });
      const data = await res.json();
      if (res.ok) {
        localStorage.setItem('token', data.token);
        setUser(data.user);
        toast.success("Verified successfully!"); // Updated to Sonner syntax
        navigate('/');
      } else {
        toast.error(data.error || "Invalid OTP"); // Updated to Sonner syntax
      }
    } catch (err) {
      console.error('OTP verify error:', err);
      toast.error("An error occurred during verification."); // Updated to Sonner syntax
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if (!email) return toast.error("Enter your email first."); // Updated to Sonner syntax
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Origin': 'http://localhost:3000',
        },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      toast.success(data.message); // Updated to Sonner syntax
    } catch (err) {
      console.error('Forgot password error:', err);
      toast.error("An error occurred while sending reset link."); // Updated to Sonner syntax
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    window.location.href = 'http://localhost:5000/auth/google';
  };

  return (
    <div className="min-h-screen pt-20 pb-12 bg-medical-light/20">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-md mx-auto">
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <Stethoscope className="h-12 w-12 text-medical" />
            </div>
            <h1 className="text-3xl font-bold text-medical-text mb-2">Welcome Back</h1>
            <p className="text-medical-text-light">Sign in to access your surgical AI dashboard</p>
          </div>

          <Card className="medical-card border-0">
            <CardHeader className="space-y-1 pb-6">
              <CardTitle className="text-2xl font-semibold text-center text-medical-text">
                Sign In
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <form onSubmit={isOtpLogin ? (isOtpSent ? handleVerifyOtp : handleSendOtp) : handleEmailLogin} className="space-y-4">
                {!isOtpLogin ? (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="email" className="text-medical-text font-medium">Email</Label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                        <Input 
                          id="email" 
                          type="email" 
                          placeholder="Enter your email" 
                          className="pl-10 border-medical-light focus:border-medical"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          required
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="password" className="text-medical-text font-medium">Password</Label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                        <Input 
                          id="password" 
                          type="password" 
                          placeholder="Enter your password" 
                          className="pl-10 border-medical-light focus:border-medical"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          required
                        />
                      </div>
                    </div>
                    <Button 
                      type="button" 
                      variant="link"
                      className="text-medical hover:text-medical/80 p-0"
                      onClick={handleForgotPassword}
                    >
                      Forgot password?
                    </Button>
                  </>
                ) : (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="phone" className="text-medical-text font-medium">Phone Number</Label>
                      <div className="relative">
                        <Phone className="absolute left-3 top-3 h-4 w-4 text-medical-text-light" />
                        <Input 
                          id="phone" 
                          type="tel" 
                          placeholder="Enter your phone number" 
                          className="pl-10 border-medical-light focus:border-medical"
                          value={phone}
                          onChange={(e) => setPhone(e.target.value)}
                          required
                        />
                      </div>
                    </div>
                    {isOtpSent && (
                      <div className="space-y-2">
                        <Label htmlFor="otp" className="text-medical-text font-medium">OTP Code</Label>
                        <Input 
                          id="otp" 
                          type="text" 
                          placeholder="Enter OTP code" 
                          className="border-medical-light focus:border-medical"
                          value={otp}
                          onChange={(e) => setOtp(e.target.value)}
                          required
                        />
                      </div>
                    )}
                  </>
                )}
                <Button 
                  type="submit" 
                  className="w-full bg-medical hover:bg-medical/90 medical-button"
                  disabled={loading}
                >
                  {loading ? 'Processing...' : (isOtpLogin ? (isOtpSent ? 'Verify OTP' : 'Send OTP') : 'Sign In')}
                </Button>
              </form>

              <div className="text-center">
                <Button 
                  variant="link"
                  onClick={() => setIsOtpLogin(!isOtpLogin)}
                  className="text-medical hover:text-medical/80"
                >
                  {isOtpLogin ? 'Sign in with email' : 'Sign in with phone OTP'}
                </Button>
              </div>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <Separator className="w-full" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-medical-text-light">Or continue with</span>
                </div>
              </div>

              <Button 
                variant="outline" 
                className="w-full border-medical-light hover:bg-medical-light/50"
                onClick={handleGoogleLogin}
              >
                <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Login with Google
              </Button>

              <div className="text-center">
                <p className="text-medical-text-light">
                  Don't have an account?{' '}
                  <Link to="/signup" className="text-medical hover:text-medical/80 font-medium">
                    Sign up
                  </Link>
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;