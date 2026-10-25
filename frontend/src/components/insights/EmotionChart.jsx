import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import {
  formatAnalyticsLabel
} from "../../utils/batchAnalytics";

function EmotionTooltip({
  active,
  payload,
  label,
  total
}) {
  if (
    !active ||
    !payload?.length
  ) {
    return null;
  }

  const count =
    payload[0]?.value || 0;

  const percentage =
    total > 0
      ? (
          (count / total) *
          100
        ).toFixed(1)
      : "0.0";

  return (
    <div className="chart-tooltip">
      <strong>
        {formatAnalyticsLabel(label)}
      </strong>

      <span>
        {count}{" "}
        {count === 1
          ? "review"
          : "reviews"}
      </span>

      <span>
        {percentage}%
      </span>
    </div>
  );
}

export default function EmotionChart({
  analytics
}) {
  if (!analytics) {
    return null;
  }

  const source =
    analytics.emotionDistribution ||
    [];

  const total =
    analytics.totalReviews || 0;

  const data = source
    .filter(
      (item) =>
        item?.label &&
        item?.count > 0
    )
    .map((item) => ({
      emotion:
        formatAnalyticsLabel(
          item.label
        ),
      rawEmotion: item.label,
      count: item.count
    }));

  const hasData =
    data.length > 0;

  const chartHeight = Math.max(
    280,
    data.length * 46
  );

  return (
    <section className="card insight-chart-card">
      <div className="insight-chart-header">
        <div>
          <span className="eyebrow">
            Emotion
          </span>

          <h3>
            Emotion Distribution
          </h3>
        </div>

        {analytics.topEmotion && (
          <div className="insight-leading-value">
            <span>
              Leading
            </span>

            <strong>
              {formatAnalyticsLabel(
                analytics.topEmotion
                  .label
              )}
            </strong>
          </div>
        )}
      </div>

      {hasData ? (
        <div
          className="emotion-chart-canvas"
          style={{
            height:
              `${chartHeight}px`
          }}
        >
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <BarChart
              data={data}
              layout="vertical"
              margin={{
                top: 10,
                right: 24,
                left: 18,
                bottom: 4
              }}
            >
              <CartesianGrid
                stroke="var(--border)"
                strokeDasharray="3 3"
                horizontal={false}
              />

              <XAxis
                type="number"
                allowDecimals={false}
                tick={{
                  fill:
                    "var(--muted)",
                  fontSize: 11
                }}
                axisLine={{
                  stroke:
                    "var(--border)"
                }}
                tickLine={false}
              />

              <YAxis
                type="category"
                dataKey="emotion"
                width={110}
                tick={{
                  fill:
                    "var(--muted)",
                  fontSize: 12
                }}
                axisLine={false}
                tickLine={false}
              />

              <Tooltip
                cursor={{
                  fill:
                    "var(--surface-muted)"
                }}
                content={
                  <EmotionTooltip
                    total={total}
                  />
                }
              />

              <Bar
                dataKey="count"
                fill="#8b5cf6"
                radius={[
                  0,
                  7,
                  7,
                  0
                ]}
                maxBarSize={28}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="insight-chart-empty">
          No emotion data is
          available.
        </div>
      )}
    </section>
  );
}