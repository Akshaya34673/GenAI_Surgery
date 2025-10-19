// src/App.jsx
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { createContext, useContext, useEffect, useState } from "react";
import Navigation from "./Navigation";
import HomePage from "./HomePage";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import DemoPage from "./DemoPage";
import NotFoundPage from "./NotFoundPage";
import ResetPassword from "./ResetPassword";
import ErrorBoundary from "./ErrorBoundary";
import { Toaster } from "@/components/ui/sonner"; // Import at the top
import HistorySidebar from "./HistorySidebar";

const queryClient = new QueryClient();

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

const AppWrapper = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get("token");
    if (token) {
      localStorage.setItem("token", token);
      navigate(location.pathname, { replace: true });
    }

    const storedToken = localStorage.getItem("token");
    if (storedToken) {
      fetch("http://localhost:5000/auth/me", {
        headers: {
          Authorization: `Bearer ${storedToken}`,
          Origin: "http://localhost:3000",
        },
      })
        .then((res) => res.json())
        .then((data) => {
          if (data.user) setUser(data.user);
          else localStorage.removeItem("token");
        })
        .catch((err) => {
          console.error("Auth error:", err);
          localStorage.removeItem("token");
        });
    }
  }, [location, navigate]);

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
    navigate("/");
  };

  return (
    <AuthContext.Provider value={{ user, setUser, logout }}>
      <ErrorBoundary>
        <Toaster /> {/* Moved to top level for early mounting */}
        <div className="scroll-smooth">
          <Navigation />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/demo" element={<DemoPage toggleSidebar={toggleSidebar} />} />
            <Route path="/reset-password/:token" element={<ResetPassword />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
          {isSidebarOpen && <HistorySidebar onClose={toggleSidebar} />}
        </div>
      </ErrorBoundary>
    </AuthContext.Provider>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <BrowserRouter>
        <AppWrapper />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;