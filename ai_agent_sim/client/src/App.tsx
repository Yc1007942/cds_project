import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import Navigation from "./components/Navigation";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import DataExplorer from "./pages/DataExplorer";
import FeatureMatrix from "./pages/FeatureMatrix";
import InferenceCore from "./pages/InferenceCore";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/data-explorer"} component={DataExplorer} />
      <Route path={"/feature-matrix"} component={FeatureMatrix} />
      <Route path={"/inference"} component={InferenceCore} />
      <Route path={"/404"} component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <div className="min-h-screen flex flex-col bg-[#04070d]">
            <Navigation />
            <Router />
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
