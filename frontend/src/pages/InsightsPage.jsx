import { useMemo } from "react";
import { Link } from "react-router-dom";

import BatchSummaryCards from "../components/batch/BatchSummaryCards";

import {
  calculateBatchAnalytics,
  formatAnalyticsLabel
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
    Number.isNaN(date.getTime())
  ) {
    return "Unknown";
  }

  return date.toLocaleString();
}

export default function InsightsPage() {
  const latestBatch =
    getLatestBatch();

  const analytics = useMemo(
    () =>
      calculateBatchAnalytics(
        latestBatch?.results || []
      ),
    [latestBatch]
  );

  function clearInsights() {
    clearLatestBatch();

    window.location.reload();
  }

  if (
    !latestBatch ||
    !latestBatch.results.length
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
    <div>
      <header className="page-header">
        <div>
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

        <div className="topbar-actions">
          <Link
            className="button button-secondary"
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

      <section className="card">
        <div className="card-header">
          <div>
            <span className="eyebrow">
              Latest source
            </span>

            <h2>
              {latestBatch.fileName}
            </h2>
          </div>

          <span className="badge">
            {analytics.totalReviews}{" "}
            reviews
          </span>
        </div>

        <div className="insight-grid">
          <div className="insight-list">
            <h3>
              Analysis details
            </h3>

            <ul>
              <li>
                Analyzed:{" "}
                {formatTimestamp(
                  latestBatch.createdAt
                )}
              </li>

              <li>
                Average rating:{" "}
                {analytics.ratedReviewCount
                  ? `${analytics.averageRating.toFixed(
                      2
                    )}/5`
                  : "-"}
              </li>
            </ul>
          </div>

          <div className="insight-list">
            <h3>
              Leading signals
            </h3>

            <ul>
              <li>
                Top emotion:{" "}
                {formatAnalyticsLabel(
                  analytics.topEmotion
                    ?.label
                )}
              </li>

              <li>
                Top positive aspect:{" "}
                {formatAnalyticsLabel(
                  analytics
                    .topPositiveAspect
                    ?.label
                )}
              </li>

              <li>
                Top negative aspect:{" "}
                {formatAnalyticsLabel(
                  analytics
                    .topNegativeAspect
                    ?.label
                )}
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section
        style={{ marginTop: "16px" }}
      >
        <BatchSummaryCards
          results={latestBatch.results}
        />
      </section>
    </div>
  );
}