import {
  AlertTriangle,
  CheckCircle2,
  Compass,
  Sparkles
} from "lucide-react";

import {
  formatAnalyticsLabel
} from "../../utils/batchAnalytics";

import "./ExecutiveSummary.css";

function getPortfolioPosition(
  analytics
) {
  const positive =
    analytics
      ?.sentimentPercentages
      ?.positive || 0;

  const negative =
    analytics
      ?.sentimentPercentages
      ?.negative || 0;

  const averageRating =
    analytics?.averageRating || 0;

  if (
    positive >= 70 &&
    averageRating >= 4
  ) {
    return {
      tone: "positive",
      title:
        "Strong guest experience",
      description:
        "The review portfolio shows a strong positive signal supported by a high average rating."
    };
  }

  if (
    negative >= 40 ||
    averageRating < 3
  ) {
    return {
      tone: "negative",
      title:
        "Guest experience requires attention",
      description:
        "The review portfolio contains a material negative signal or a below-average rating profile."
    };
  }

  return {
    tone: "mixed",
    title:
      "Mixed guest experience",
    description:
      "The review portfolio shows a balanced or mixed experience with identifiable strengths and improvement opportunities."
  };
}

function buildExecutiveNarrative(
  analytics
) {
  const total =
    analytics?.totalReviews || 0;

  const positive =
    analytics
      ?.sentimentPercentages
      ?.positive || 0;

  const neutral =
    analytics
      ?.sentimentPercentages
      ?.neutral || 0;

  const negative =
    analytics
      ?.sentimentPercentages
      ?.negative || 0;

  const rating =
    analytics?.ratedReviewCount
      ? `${analytics.averageRating.toFixed(
          2
        )}/5`
      : "not available";

  const topEmotion =
    formatAnalyticsLabel(
      analytics?.topEmotion?.label
    );

  return (
    `Across ${total} analyzed ` +
    `${total === 1 ? "review" : "reviews"}, ` +
    `${positive}% were positive, ` +
    `${neutral}% were neutral, and ` +
    `${negative}% were negative. ` +
    `The average predicted rating was ` +
    `${rating}, with ${topEmotion} as ` +
    `the most frequently detected emotion.`
  );
}

function getStrength(
  analytics
) {
  const topAspect =
    analytics
      ?.topPositiveAspect;

  if (!topAspect) {
    return {
      title:
        "No dominant strength identified",
      description:
        "The analyzed reviews do not contain enough positive aspect evidence to identify a leading strength."
    };
  }

  return {
    title:
      formatAnalyticsLabel(
        topAspect.label
      ),
    description:
      `${topAspect.count} positive ` +
      `${topAspect.count === 1
        ? "mention"
        : "mentions"} made this the ` +
      `strongest positive aspect in the latest batch.`
  };
}

function getRisk(
  analytics
) {
  const topAspect =
    analytics
      ?.topNegativeAspect;

  if (!topAspect) {
    return {
      title:
        "No dominant risk identified",
      description:
        "The analyzed reviews do not contain a recurring negative aspect signal."
    };
  }

  return {
    title:
      formatAnalyticsLabel(
        topAspect.label
      ),
    description:
      `${topAspect.count} negative ` +
      `${topAspect.count === 1
        ? "mention"
        : "mentions"} made this the ` +
      `leading improvement opportunity in the latest batch.`
  };
}

function getRecommendedFocus(
  analytics
) {
  const negativeAspect =
    analytics
      ?.topNegativeAspect;

  const positiveAspect =
    analytics
      ?.topPositiveAspect;

  if (negativeAspect) {
    const negativeLabel =
      formatAnalyticsLabel(
        negativeAspect.label
      );

    return {
      title:
        `Prioritize ${negativeLabel}`,
      description:
        `Review the evidence and recurring issues associated with ${negativeLabel}. Preserve the strongest positive experiences while addressing this leading negative signal.`
    };
  }

  if (positiveAspect) {
    const positiveLabel =
      formatAnalyticsLabel(
        positiveAspect.label
      );

    return {
      title:
        `Protect ${positiveLabel}`,
      description:
        `Maintain the operating practices contributing to positive feedback about ${positiveLabel}, and continue monitoring future batches for emerging concerns.`
    };
  }

  return {
    title:
      "Continue monitoring",
    description:
      "The latest batch does not contain a dominant aspect signal. Continue reviewing future batches before prioritizing an operational response."
  };
}

export default function ExecutiveSummary({
  analytics
}) {
  if (
    !analytics ||
    !analytics.totalReviews
  ) {
    return null;
  }

  const position =
    getPortfolioPosition(
      analytics
    );

  const strength =
    getStrength(
      analytics
    );

  const risk =
    getRisk(
      analytics
    );

  const recommendedFocus =
    getRecommendedFocus(
      analytics
    );

  const narrative =
    buildExecutiveNarrative(
      analytics
    );

  return (
    <section className="card executive-summary-card">
      <div className="executive-summary-header">
        <div className="executive-summary-heading">
          <span className="executive-summary-icon">
            <Sparkles
              size={20}
              aria-hidden="true"
            />
          </span>

          <div>
            <span className="eyebrow">
              Management overview
            </span>

            <h3>
              Executive Summary
            </h3>
          </div>
        </div>

        <span
          className={
            `executive-position-badge ` +
            `executive-position-${position.tone}`
          }
        >
          {position.title}
        </span>
      </div>

      <div className="executive-overview">
        <strong>
          {position.title}
        </strong>

        <p>
          {position.description}
        </p>

        <p className="executive-narrative">
          {narrative}
        </p>
      </div>

      <div className="executive-signal-grid">
        <article className="executive-signal executive-signal-positive">
          <span className="executive-signal-icon">
            <CheckCircle2
              size={19}
              aria-hidden="true"
            />
          </span>

          <div>
            <span className="eyebrow">
              Leading strength
            </span>

            <h4>
              {strength.title}
            </h4>

            <p>
              {strength.description}
            </p>
          </div>
        </article>

        <article className="executive-signal executive-signal-negative">
          <span className="executive-signal-icon">
            <AlertTriangle
              size={19}
              aria-hidden="true"
            />
          </span>

          <div>
            <span className="eyebrow">
              Improvement opportunity
            </span>

            <h4>
              {risk.title}
            </h4>

            <p>
              {risk.description}
            </p>
          </div>
        </article>

        <article className="executive-signal executive-signal-focus">
          <span className="executive-signal-icon">
            <Compass
              size={19}
              aria-hidden="true"
            />
          </span>

          <div>
            <span className="eyebrow">
              Recommended focus
            </span>

            <h4>
              {recommendedFocus.title}
            </h4>

            <p>
              {
                recommendedFocus
                  .description
              }
            </p>
          </div>
        </article>
      </div>

      <p className="executive-summary-note">
        This summary is generated
        deterministically from the latest
        batch analytics. Review-level
        evidence should be examined before
        operational decisions are made.
      </p>
    </section>
  );
}