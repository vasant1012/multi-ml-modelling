import { useState } from "react";

import { analyzeBatch }
  from "../api/reviewApi";

export default function useBatchAnalysis() {
  const [status, setStatus] =
    useState("idle");

  const [results, setResults] =
    useState([]);

  const [summary, setSummary] =
    useState(null);

  const [error, setError] =
    useState(null);

  async function run(rows) {
    if (!rows?.length) {
      return null;
    }

    setStatus("loading");
    setError(null);

    try {
      const reviews = rows.map(
        (row) => row.review_text
      );

      const response =
        await analyzeBatch(reviews);

      const batchResults =
        response.results || [];

      setResults(batchResults);

      setSummary({
        totalReviews:
          batchResults.length
      });

      setStatus("success");

      return batchResults;
    } catch (e) {
      setError(
        e.response?.data?.detail ||
          e.message ||
          "Batch analysis failed."
      );

      setStatus("error");

      return null;
    }
  }

  function reset() {
    setStatus("idle");
    setResults([]);
    setSummary(null);
    setError(null);
  }

  return {
    status,
    results,
    summary,
    error,
    run,
    reset
  };
}