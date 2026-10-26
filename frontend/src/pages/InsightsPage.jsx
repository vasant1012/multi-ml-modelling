import {
  useMemo,
  useState
} from "react";

import {
  Link
} from "react-router-dom";

import BatchSummaryCards from "../components/batch/BatchSummaryCards";

import SentimentChart from "../components/insights/SentimentChart";
import RatingChart from "../components/insights/RatingChart";
import EmotionChart from "../components/insights/EmotionChart";
import AspectChart from "../components/insights/AspectChart";
import ExecutiveSummary from "../components/insights/ExecutiveSummary";

import {
  calculateBatchAnalytics
} from "../utils/batchAnalytics";

import {
  clearLatestBatch,
  getLatestBatch
} from "../utils/storage";

function formatTimestamp(value) {
  if (!value) {
    return "Unknown";
  }

  const date = new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return "Unknown";
  }

  return date.toLocaleString();
}

export default function InsightsPage() {
  const [
    latestBatch,
    setLatestBatch
  ] = useState(() =>
    getLatestBatch()
  );

  const analytics = useMemo(
    () =>
      calculateBatchAnalytics(
        latestBatch?.results || []
      ),
    [latestBatch]
  );

  function clearInsights() {
    clearLatestBatch();
    setLatestBatch(null);
  }

  if (
    !latestBatch ||
    !latestBatch.results?.length
  ) {
    return (
      <div className="placeholder-page">
        <div>
          <span className="eyebrow">
            Portfolio analytics
          </span>

          <h2>
            No batch insights available
          </h2>

          <p>
            Run a batch analysis to
            generate portfolio-level
            sentiment, rating, emotion,
            and aspect insights.
          </p>

          <Link
            className="button button-primary"
            to="/batch"
          >
            Go to Batch Analysis
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="insights-page">
      <header className="insights-compact-header">
        <div className="insights-heading">
          <span className="eyebrow">
            Portfolio analytics
          </span>

          <h2>
            Portfolio Insights
          </h2>

          <p>
            Latest completed batch
            analysis across{" "}
            {analytics.totalReviews}{" "}
            reviews.
          </p>
        </div>

        <div className="insights-source-summary">
          <span className="eyebrow">
            Latest source
          </span>

          <strong>
            {latestBatch.fileName}
          </strong>

          <div className="insights-source-meta">
            <span>
              {analytics.totalReviews}{" "}
              reviews
            </span>

            <span
              className="insights-meta-divider"
              aria-hidden="true"
            >
              •
            </span>

            <span>
              Analyzed{" "}
              {formatTimestamp(
                latestBatch.createdAt
              )}
            </span>
          </div>
        </div>

        <div className="insights-header-actions">
          <Link
            className="button button-primary"
            to="/batch"
          >
            Open Batch Analysis
          </Link>

          <button
            type="button"
            className="button button-secondary"
            onClick={clearInsights}
          >
            Clear Insights
          </button>
        </div>
      </header>

      <section className="insights-kpi-section">
        <BatchSummaryCards
          results={
            latestBatch.results
          }
        />
      </section>

      <section className="insights-chart-grid">
        <SentimentChart
          analytics={analytics}
        />

        <RatingChart
          analytics={analytics}
        />
      </section>

      <section className="insights-chart-grid insights-secondary-charts">
        <EmotionChart
          analytics={analytics}
        />

        <AspectChart
          analytics={analytics}
        />
      </section>

      <ExecutiveSummary
        analytics={analytics}
      />
    </div>
  );
}