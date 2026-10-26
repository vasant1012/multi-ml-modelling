import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import {
  formatAnalyticsLabel
} from "../../utils/batchAnalytics";

const MAX_ASPECTS = 8;

function AspectTooltip({
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

  const positive =
    payload.find(
      (item) =>
        item.dataKey ===
        "positive"
    )?.value || 0;

  const negative =
    payload.find(
      (item) =>
        item.dataKey ===
        "negative"
    )?.value || 0;

  return (
    <div className="chart-tooltip">
      <strong>
        {label}
      </strong>

      <span>
        Positive: {positive}
      </span>

      <span>
        Negative: {negative}
      </span>
    </div>
  );
}

function buildAspectData(
  analytics
) {
  const positiveMap =
    new Map(
      (
        analytics
          ?.positiveAspects || []
      ).map((item) => [
        item.label,
        item.count
      ])
    );

  const negativeMap =
    new Map(
      (
        analytics
          ?.negativeAspects || []
      ).map((item) => [
        item.label,
        item.count
      ])
    );

  const labels =
    new Set([
      ...positiveMap.keys(),
      ...negativeMap.keys()
    ]);

  return Array.from(labels)
    .map((label) => {
      const positive =
        positiveMap.get(label) || 0;

      const negative =
        negativeMap.get(label) || 0;

      return {
        aspect:
          formatAnalyticsLabel(
            label
          ),
        rawAspect: label,
        positive,
        negative,
        total:
          positive + negative
      };
    })
    .sort(
      (a, b) =>
        b.total - a.total
    )
    .slice(
      0,
      MAX_ASPECTS
    );
}

export default function AspectChart({
  analytics
}) {
  if (!analytics) {
    return null;
  }

  const data =
    buildAspectData(
      analytics
    );

  const hasData =
    data.length > 0;

  const chartHeight =
    Math.max(
      300,
      data.length * 50
    );

  return (
    <section className="card insight-chart-card">
      <div className="insight-chart-header">
        <div>
          <span className="eyebrow">
            Aspects
          </span>

          <h3>
            Aspect Intelligence
          </h3>
        </div>

        <span className="insight-chart-total">
          Top {data.length} aspects
        </span>
      </div>

      {hasData ? (
        <div
          className="aspect-chart-canvas"
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
                top: 8,
                right: 24,
                left: 22,
                bottom: 8
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
                dataKey="aspect"
                width={120}
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
                  <AspectTooltip />
                }
              />

              <Legend
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

              <Bar
                dataKey="positive"
                name="Positive"
                fill="#22c55e"
                radius={[
                  0,
                  6,
                  6,
                  0
                ]}
                maxBarSize={18}
              />

              <Bar
                dataKey="negative"
                name="Negative"
                fill="#ef4444"
                radius={[
                  0,
                  6,
                  6,
                  0
                ]}
                maxBarSize={18}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="insight-chart-empty">
          No aspect data is
          available.
        </div>
      )}
    </section>
  );
}