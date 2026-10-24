import { useMemo } from "react";

export default function BatchSummaryCards({
  results = []
}) {
  const summary = useMemo(() => {
    if (!results.length) {
      return null;
    }

    const sentimentCounts = {
      positive: 0,
      neutral: 0,
      negative: 0
    };

    const emotionCounts = {};
    const positiveAspects = {};
    const negativeAspects = {};

    let ratingSum = 0;

    results.forEach((result) => {
      const sentiment =
        result.sentiment?.label?.toLowerCase();

      if (
        sentiment &&
        sentimentCounts[sentiment] !== undefined
      ) {
        sentimentCounts[sentiment]++;
      }

      ratingSum +=
        result.rating?.prediction || 0;

      const emotion =
        result.emotion?.dominant;

      if (emotion) {
        emotionCounts[emotion] =
          (emotionCounts[emotion] || 0) + 1;
      }

      (result.aspects || []).forEach(
        (aspect) => {
          if (
            aspect.sentiment
              ?.toLowerCase() ===
            "positive"
          ) {
            positiveAspects[
              aspect.aspect
            ] =
              (positiveAspects[
                aspect.aspect
              ] || 0) + 1;
          }

          if (
            aspect.sentiment
              ?.toLowerCase() ===
            "negative"
          ) {
            negativeAspects[
              aspect.aspect
            ] =
              (negativeAspects[
                aspect.aspect
              ] || 0) + 1;
          }
        }
      );
    });

    const total = results.length;

    const averageRating =
      ratingSum / total;

    const getTopKey = (obj) => {
      const entries =
        Object.entries(obj);

      if (!entries.length) {
        return "-";
      }

      return entries.sort(
        (a, b) => b[1] - a[1]
      )[0][0];
    };

    return {
      totalReviews: total,

      positivePercent:
        (
          (sentimentCounts.positive /
            total) *
          100
        ).toFixed(1),

      neutralPercent:
        (
          (sentimentCounts.neutral /
            total) *
          100
        ).toFixed(1),

      negativePercent:
        (
          (sentimentCounts.negative /
            total) *
          100
        ).toFixed(1),

      averageRating:
        averageRating.toFixed(2),

      topEmotion:
        getTopKey(emotionCounts),

      topPositiveAspect:
        getTopKey(positiveAspects),

      topNegativeAspect:
        getTopKey(negativeAspects)
    };
  }, [results]);

  if (!summary) {
    return null;
  }

  return (
    <div className="summary-card-grid">
      <div className="card">
        <h4>Total Reviews</h4>
        <strong>
          {summary.totalReviews}
        </strong>
      </div>

      <div className="card">
        <h4>Average Rating</h4>
        <strong>
          {summary.averageRating}/5
        </strong>
      </div>

      <div className="card">
        <h4>Positive</h4>
        <strong>
          {summary.positivePercent}%
        </strong>
      </div>

      <div className="card">
        <h4>Neutral</h4>
        <strong>
          {summary.neutralPercent}%
        </strong>
      </div>

      <div className="card">
        <h4>Negative</h4>
        <strong>
          {summary.negativePercent}%
        </strong>
      </div>

      <div className="card">
        <h4>Top Emotion</h4>
        <strong>
          {summary.topEmotion}
        </strong>
      </div>

      <div className="card">
        <h4>Top Positive Aspect</h4>
        <strong>
          {summary.topPositiveAspect}
        </strong>
      </div>

      <div className="card">
        <h4>Top Negative Aspect</h4>
        <strong>
          {summary.topNegativeAspect}
        </strong>
      </div>
    </div>
  );
}