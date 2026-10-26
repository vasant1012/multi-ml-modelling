import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function RatingTooltip({
  active,
  payload,
  label
}) {
  if (
    !active ||
    !payload?.length
  ) {
    return null;
  }

  return (
    <div className="chart-tooltip">
      <strong>
        {label} Star
        {Number(label) !== 1
          ? "s"
          : ""}
      </strong>

      <span>
        {payload[0].value} reviews
      </span>
    </div>
  );
}

export default function RatingChart({
  analytics
}) {
  if (!analytics) {
    return null;
  }

  const source =
    analytics.ratingDistribution ||
    [];

  const data = [1, 2, 3, 4, 5].map(
    (rating) => {
      const match = source.find(
        (item) =>
          Number(item.rating) ===
          rating
      );

      return {
        rating,
        count:
          match?.count || 0
      };
    }
  );

  const hasData =
    data.some(
      (item) => item.count > 0
    );

  return (
    <section className="card insight-chart-card">
      <div className="insight-chart-header">
        <div>
          <span className="eyebrow">
            Rating
          </span>

          <h3>
            Rating Distribution
          </h3>
        </div>

        <div className="rating-average">
          <strong>
            {analytics.ratedReviewCount
              ? analytics.averageRating.toFixed(
                  2
                )
              : "-"}
          </strong>

          <span>/5 average</span>
        </div>
      </div>

      {hasData ? (
        <div className="rating-chart-canvas">
          <ResponsiveContainer
            width="100%"
            height="100%"
          >
            <BarChart
              data={data}
              margin={{
                top: 14,
                right: 12,
                left: -18,
                bottom: 0
              }}
            >
              <CartesianGrid
                stroke="var(--border)"
                strokeDasharray="3 3"
                vertical={false}
              />

              <XAxis
                dataKey="rating"
                tick={{
                  fill:
                    "var(--muted)",
                  fontSize: 12
                }}
                axisLine={{
                  stroke:
                    "var(--border)"
                }}
                tickLine={false}
                tickFormatter={(
                  value
                ) => `${value} ★`}
              />

              <YAxis
                allowDecimals={false}
                tick={{
                  fill:
                    "var(--muted)",
                  fontSize: 11
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
                  <RatingTooltip />
                }
              />

              <Bar
                dataKey="count"
                fill="var(--primary)"
                radius={[
                  7,
                  7,
                  0,
                  0
                ]}
                maxBarSize={64}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="insight-chart-empty">
          No rating data is
          available.
        </div>
      )}
    </section>
  );
}