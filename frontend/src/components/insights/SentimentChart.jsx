import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip
} from "recharts";

const COLORS = {
  Positive: "#22c55e",
  Neutral: "#f59e0b",
  Negative: "#ef4444"
};

function ChartTooltip({
  active,
  payload,
  total
}) {
  if (
    !active ||
    !payload?.length
  ) {
    return null;
  }

  const item = payload[0]?.payload;

  if (!item) {
    return null;
  }

  const percentage =
    total > 0
      ? (
          (item.value / total) *
          100
        ).toFixed(1)
      : "0.0";

  return (
    <div className="chart-tooltip">
      <strong>{item.name}</strong>

      <span>
        {item.value} reviews
      </span>

      <span>
        {percentage}%
      </span>
    </div>
  );
}

export default function SentimentChart({
  analytics
}) {
  if (!analytics) {
    return null;
  }

  const data = [
    {
      name: "Positive",
      value:
        analytics.sentimentCounts
          ?.positive || 0
    },
    {
      name: "Neutral",
      value:
        analytics.sentimentCounts
          ?.neutral || 0
    },
    {
      name: "Negative",
      value:
        analytics.sentimentCounts
          ?.negative || 0
    }
  ];

  const total = data.reduce(
    (sum, item) =>
      sum + item.value,
    0
  );

  return (
    <section className="card insight-chart-card">
      <div className="insight-chart-header">
        <div>
          <span className="eyebrow">
            Sentiment
          </span>

          <h3>
            Sentiment Distribution
          </h3>
        </div>

        <span className="insight-chart-total">
          {total} reviews
        </span>
      </div>

      {total > 0 ? (
        <div className="sentiment-chart-layout">
          <div className="insight-chart-canvas">
            <ResponsiveContainer
              width="100%"
              height="100%"
            >
              <PieChart>
                <Pie
                  data={data}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius="55%"
                  outerRadius="78%"
                  paddingAngle={3}
                  stroke="none"
                >
                  {data.map((item) => (
                    <Cell
                      key={item.name}
                      fill={
                        COLORS[
                          item.name
                        ]
                      }
                    />
                  ))}
                </Pie>

                <Tooltip
                  content={
                    <ChartTooltip
                      total={total}
                    />
                  }
                />

                <Legend
                  verticalAlign="bottom"
                  iconType="circle"
                  formatter={(value) => (
                    <span
                      style={{
                        color:
                          "var(--text)"
                      }}
                    >
                      {value}
                    </span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>

            <div className="donut-center">
              <strong>
                {analytics
                  .sentimentPercentages
                  ?.positive || 0}
                %
              </strong>

              <span>Positive</span>
            </div>
          </div>

          <div className="sentiment-breakdown">
            {data.map((item) => {
              const percentage =
                total > 0
                  ? (
                      (item.value /
                        total) *
                      100
                    ).toFixed(1)
                  : "0.0";

              return (
                <div
                  className="sentiment-breakdown-row"
                  key={item.name}
                >
                  <span
                    className="sentiment-dot"
                    style={{
                      background:
                        COLORS[
                          item.name
                        ]
                    }}
                  />

                  <span>
                    {item.name}
                  </span>

                  <strong>
                    {percentage}%
                  </strong>

                  <small>
                    {item.value}
                  </small>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="insight-chart-empty">
          No sentiment data is
          available.
        </div>
      )}
    </section>
  );
}