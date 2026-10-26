import { useState } from "react";
import { analyzeReview } from "../api/reviewApi";

export default function useReviewAnalysis() {
  const [status, setStatus] = useState("idle");
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  async function run(text) {
    setStatus("loading");
    setError(null);

    try {
      const data = await analyzeReview(text);

      setAnalysis(data);
      setStatus("success");

      return data;
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        e.message ||
        "Analysis failed."
      );

      setStatus("error");
      return null;
    }
  }

  function loadAnalysis(existingAnalysis) {
    if (!existingAnalysis) return;

    setAnalysis(existingAnalysis);
    setError(null);
    setStatus("success");
  }

  function reset() {
    setStatus("idle");
    setAnalysis(null);
    setError(null);
  }

  return {
    status,
    analysis,
    error,
    run,
    loadAnalysis,
    reset
  };
}