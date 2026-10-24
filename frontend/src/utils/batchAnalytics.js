const SENTIMENT_KEYS = [
  "positive",
  "neutral",
  "negative"
];

const normalizeLabel = (value) =>
  String(value || "")
    .trim()
    .toLowerCase();

const incrementCount = (
  collection,
  key
) => {
  if (!key) {
    return;
  }

  collection[key] =
    (collection[key] || 0) + 1;
};

const sortCounts = (collection) =>
  Object.entries(collection)
    .map(([label, count]) => ({
      label,
      count
    }))
    .sort((a, b) => {
      if (b.count !== a.count) {
        return b.count - a.count;
      }

      return a.label.localeCompare(
        b.label
      );
    });

const getTopItem = (items) => {
  if (!items.length) {
    return null;
  }

  return items[0];
};

const calculatePercent = (
  count,
  total
) => {
  if (!total) {
    return 0;
  }

  return Number(
    ((count / total) * 100).toFixed(1)
  );
};

export function calculateBatchAnalytics(
  results = []
) {
  const safeResults = Array.isArray(results)
    ? results
    : [];

  const sentimentCounts = {
    positive: 0,
    neutral: 0,
    negative: 0
  };

  const ratingCounts = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0
  };

  const emotionCounts = {};
  const aspectMentions = {};
  const positiveAspectCounts = {};
  const neutralAspectCounts = {};
  const negativeAspectCounts = {};

  let ratingSum = 0;
  let ratedReviewCount = 0;

  safeResults.forEach((result) => {
    const sentiment = normalizeLabel(
      result?.sentiment?.label
    );

    if (
      SENTIMENT_KEYS.includes(sentiment)
    ) {
      sentimentCounts[sentiment] += 1;
    }

    const rating = Number(
      result?.rating?.prediction
    );

    if (
      Number.isFinite(rating) &&
      rating >= 1 &&
      rating <= 5
    ) {
      ratingSum += rating;
      ratedReviewCount += 1;

      const ratingKey =
        Math.round(rating);

      if (ratingCounts[ratingKey] !== undefined) {
        ratingCounts[ratingKey] += 1;
      }
    }

    const emotion = normalizeLabel(
      result?.emotion?.dominant
    );

    incrementCount(
      emotionCounts,
      emotion
    );

    const aspects = Array.isArray(
      result?.aspects
    )
      ? result.aspects
      : [];

    aspects.forEach((aspect) => {
      const aspectName = normalizeLabel(
        aspect?.aspect
      );

      const aspectSentiment =
        normalizeLabel(
          aspect?.sentiment
        );

      if (!aspectName) {
        return;
      }

      incrementCount(
        aspectMentions,
        aspectName
      );

      if (
        aspectSentiment === "positive"
      ) {
        incrementCount(
          positiveAspectCounts,
          aspectName
        );
      }

      if (
        aspectSentiment === "neutral"
      ) {
        incrementCount(
          neutralAspectCounts,
          aspectName
        );
      }

      if (
        aspectSentiment === "negative"
      ) {
        incrementCount(
          negativeAspectCounts,
          aspectName
        );
      }
    });
  });

  const totalReviews =
    safeResults.length;

  const averageRating =
    ratedReviewCount > 0
      ? Number(
          (
            ratingSum /
            ratedReviewCount
          ).toFixed(2)
        )
      : 0;

  const emotionDistribution =
    sortCounts(emotionCounts);

  const aspectDistribution =
    sortCounts(aspectMentions);

  const positiveAspects =
    sortCounts(positiveAspectCounts);

  const neutralAspects =
    sortCounts(neutralAspectCounts);

  const negativeAspects =
    sortCounts(negativeAspectCounts);

  return {
    totalReviews,
    averageRating,
    ratedReviewCount,

    sentimentCounts,

    sentimentPercentages: {
      positive: calculatePercent(
        sentimentCounts.positive,
        totalReviews
      ),
      neutral: calculatePercent(
        sentimentCounts.neutral,
        totalReviews
      ),
      negative: calculatePercent(
        sentimentCounts.negative,
        totalReviews
      )
    },

    ratingDistribution:
      Object.entries(ratingCounts).map(
        ([rating, count]) => ({
          rating: Number(rating),
          count
        })
      ),

    emotionDistribution,
    aspectDistribution,
    positiveAspects,
    neutralAspects,
    negativeAspects,

    topEmotion:
      getTopItem(
        emotionDistribution
      ),

    topPositiveAspect:
      getTopItem(
        positiveAspects
      ),

    topNegativeAspect:
      getTopItem(
        negativeAspects
      )
  };
}

export function formatAnalyticsLabel(
  value
) {
  const normalized = String(
    value || ""
  ).trim();

  if (!normalized) {
    return "-";
  }

  return normalized
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}