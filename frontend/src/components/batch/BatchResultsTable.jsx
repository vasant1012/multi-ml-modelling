import {
  useEffect,
  useMemo,
  useState
} from "react";

const PAGE_SIZE = 25;

export default function BatchResultsTable({
  results = []
}) {
  const [page, setPage] = useState(1);

  const totalPages = Math.max(
    1,
    Math.ceil(
      results.length / PAGE_SIZE
    )
  );

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);

  const currentRows = useMemo(() => {
    const start =
      (page - 1) * PAGE_SIZE;

    return results.slice(
      start,
      start + PAGE_SIZE
    );
  }, [results, page]);

  function getPrimaryAspect(result) {
    if (!result?.aspects?.length) {
      return "-";
    }

    return (
      result.aspects[0]?.aspect ||
      "-"
    );
  }

  function getDisplayValue(
    value,
    fallback = "-"
  ) {
    if (
      value === null ||
      value === undefined ||
      value === ""
    ) {
      return fallback;
    }

    return value;
  }

  return (
    <div className="card batch-results-card">
      <div className="batch-results-header">
        <div className="batch-results-title">
          <h3>
            Batch Analysis Results
          </h3>

          <span className="batch-results-count">
            {results.length}{" "}
            {results.length === 1
              ? "Review"
              : "Reviews"}
          </span>
        </div>

        {totalPages > 1 && (
          <span className="batch-results-page-status">
            Page {page} of{" "}
            {totalPages}
          </span>
        )}
      </div>

      <div className="batch-table-wrapper">
        <table className="batch-results-table">
          <colgroup>
            <col className="batch-col-index" />
            <col className="batch-col-review" />
            <col className="batch-col-sentiment" />
            <col className="batch-col-rating" />
            <col className="batch-col-emotion" />
            <col className="batch-col-aspect" />
            <col className="batch-col-quality" />
          </colgroup>

          <thead>
            <tr>
              <th scope="col">#</th>

              <th scope="col">
                Review
              </th>

              <th scope="col">
                Sentiment
              </th>

              <th scope="col">
                Rating
              </th>

              <th scope="col">
                Emotion
              </th>

              <th scope="col">
                Primary Aspect
              </th>

              <th scope="col">
                Quality
              </th>
            </tr>
          </thead>

          <tbody>
            {currentRows.map(
              (result, index) => {
                const rowNumber =
                  (page - 1) *
                    PAGE_SIZE +
                  index +
                  1;

                const sentiment =
                  getDisplayValue(
                    result?.sentiment
                      ?.label
                  );

                const rating =
                  result?.rating
                    ?.prediction;

                const emotion =
                  getDisplayValue(
                    result?.emotion
                      ?.dominant
                  );

                const quality =
                  getDisplayValue(
                    result?.quality
                      ?.status
                  );

                return (
                  <tr
                    key={`${page}-${rowNumber}`}
                  >
                    <td className="batch-row-number">
                      {rowNumber}
                    </td>

                    <td className="batch-review-cell">
                      {
                        result.review_text
                      }
                    </td>

                    <td>
                      <span
                        className={`sentiment-badge sentiment-${String(
                          sentiment
                        ).toLowerCase()}`}
                      >
                        {sentiment}
                      </span>
                    </td>

                    <td className="batch-rating-cell">
                      {rating !== null &&
                      rating !==
                        undefined
                        ? `${rating}/5`
                        : "-"}
                    </td>

                    <td>
                      {emotion}
                    </td>

                    <td>
                      {getPrimaryAspect(
                        result
                      )}
                    </td>

                    <td>
                      {quality}
                    </td>
                  </tr>
                );
              }
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            type="button"
            className="button button-secondary"
            disabled={page === 1}
            onClick={() =>
              setPage((current) =>
                Math.max(
                  1,
                  current - 1
                )
              )
            }
          >
            Previous
          </button>

          <span>
            Page {page} of{" "}
            {totalPages}
          </span>

          <button
            type="button"
            className="button button-secondary"
            disabled={
              page === totalPages
            }
            onClick={() =>
              setPage((current) =>
                Math.min(
                  totalPages,
                  current + 1
                )
              )
            }
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}