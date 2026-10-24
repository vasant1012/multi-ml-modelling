import { useMemo } from "react";

import {
  calculateBatchAnalytics,
  formatAnalyticsLabel
} from "../../utils/batchAnalytics";

export default function BatchSummaryCards({
  results = []
}) {
  const analytics = useMemo(
    () =>
      calculateBatchAnalytics(
        results
      ),
    [results]
  );

  if (!analytics.totalReviews) {
    return null;
  }

  const topEmotion =
    analytics.topEmotion?.label;

  const topPositiveAspect =
    analytics.topPositiveAspect
      ?.label;

  const topNegativeAspect =
    analytics.topNegativeAspect
      ?.label;

  return (
    <div className="summary-card-grid">
      <div className="card">
        <h4>Total Reviews</h4>

        <strong>
          {analytics.totalReviews}
        </strong>
      </div>

      <div className="card">
        <h4>Average Rating</h4>

        <strong>
          {analytics.ratedReviewCount
            ? `${analytics.averageRating.toFixed(
                2
              )}/5`
            : "-"}
        </strong>
      </div>

      <div className="card">
        <h4>Positive</h4>

        <strong>
          {
            analytics
              .sentimentPercentages
              .positive
          }
          %
        </strong>
      </div>

      <div className="card">
        <h4>Neutral</h4>

        <strong>
          {
            analytics
              .sentimentPercentages
              .neutral
          }
          %
        </strong>
      </div>

      <div className="card">
        <h4>Negative</h4>

        <strong>
          {
            analytics
              .sentimentPercentages
              .negative
          }
          %
        </strong>
      </div>

      <div className="card">
        <h4>Top Emotion</h4>

        <strong>
          {formatAnalyticsLabel(
            topEmotion
          )}
        </strong>
      </div>

      <div className="card">
        <h4>
          Top Positive Aspect
        </h4>

        <strong>
          {formatAnalyticsLabel(
            topPositiveAspect
          )}
        </strong>
      </div>

      <div className="card">
        <h4>
          Top Negative Aspect
        </h4>

        <strong>
          {formatAnalyticsLabel(
            topNegativeAspect
          )}
        </strong>
      </div>
    </div>
  );
}