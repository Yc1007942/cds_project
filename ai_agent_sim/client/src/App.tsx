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
import GraphsFindings from "./pages/GraphsFindings";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/data-explorer"} component={DataExplorer} />
      <Route path={"/feature-matrix"} component={FeatureMatrix} />
      <Route path={"/inference"} component={InferenceCore} />
      <Route path={"/graphs-findings"} component={GraphsFindings} />
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
          <div className="h-screen flex flex-col bg-[#04070d] overflow-hidden">
            <Navigation />
            <div className="flex-1 overflow-hidden flex flex-col">
              <Router />
            </div>
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
