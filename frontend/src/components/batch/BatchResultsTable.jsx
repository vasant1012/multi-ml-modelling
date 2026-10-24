import { useMemo, useState } from "react";

const PAGE_SIZE = 25;

export default function BatchResultsTable({
  results = []
}) {
  const [page, setPage] = useState(1);

  const totalPages = Math.max(
    1,
    Math.ceil(results.length / PAGE_SIZE)
  );

  const currentRows = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;

    return results.slice(
      start,
      start + PAGE_SIZE
    );
  }, [results, page]);

  function getPrimaryAspect(result) {
    if (!result?.aspects?.length) {
      return "-";
    }

    return result.aspects[0].aspect;
  }

  return (
    <div className="card batch-results-card">
      <div className="table-header">
        <h3>Batch Analysis Results</h3>

        <span>
          {results.length} Reviews
        </span>
      </div>

      <div className="table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Review</th>
              <th>Sentiment</th>
              <th>Rating</th>
              <th>Emotion</th>
              <th>Primary Aspect</th>
              <th>Quality</th>
            </tr>
          </thead>

          <tbody>
            {currentRows.map(
              (result, index) => (
                <tr
                  key={`${page}-${index}`}
                >
                  <td>
                    {(page - 1) *
                      PAGE_SIZE +
                      index +
                      1}
                  </td>

                  <td className="review-cell">
                    {result.review_text}
                  </td>

                  <td>
                    <span
                      className={`sentiment-badge sentiment-${(
                        result.sentiment
                          ?.label || ""
                      ).toLowerCase()}`}
                    >
                      {
                        result.sentiment
                          ?.label
                      }
                    </span>
                  </td>

                  <td>
                    {
                      result.rating
                        ?.prediction
                    }
                    /5
                  </td>

                  <td>
                    {
                      result.emotion
                        ?.dominant
                    }
                  </td>

                  <td>
                    {getPrimaryAspect(
                      result
                    )}
                  </td>

                  <td>
                    {
                      result.quality
                        ?.status
                    }
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button
            disabled={page === 1}
            onClick={() =>
              setPage((p) => p - 1)
            }
          >
            Previous
          </button>

          <span>
            Page {page} of{" "}
            {totalPages}
          </span>

          <button
            disabled={
              page === totalPages
            }
            onClick={() =>
              setPage((p) => p + 1)
            }
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}