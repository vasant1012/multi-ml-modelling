import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";
import ReviewAnalysisPage from "./pages/ReviewAnalysisPage";
import BatchAnalysisPage from "./pages/BatchAnalysisPage";
import InsightsPage from "./pages/InsightsPage";
import HistoryPage from "./pages/HistoryPage";
export default function App(){return <AppShell><Routes><Route path="/analyze" element={<ReviewAnalysisPage/>}/><Route path="/batch" element={<BatchAnalysisPage/>}/><Route path="/insights" element={<InsightsPage/>}/><Route path="/history" element={<HistoryPage/>}/><Route path="*" element={<Navigate to="/analyze" replace/>}/></Routes></AppShell>}
