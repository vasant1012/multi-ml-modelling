export default function BatchDownloadActions({
  results = []
}) {
  function downloadCsv() {
    if (!results.length) {
      return;
    }

    const headers = [
      "review_text",
      "sentiment",
      "sentiment_confidence",
      "rating",
      "rating_confidence",
      "emotion",
      "emotion_confidence",
      "primary_aspect",
      "primary_aspect_sentiment",
      "quality_status"
    ];

    const rows = results.map(
      (result) => {
        const primaryAspect =
          result.aspects?.[0] || {};

        return [
          escapeCsv(
            result.review_text
          ),

          escapeCsv(
            result.sentiment?.label || ""
          ),

          result.sentiment
            ?.confidence ?? "",

          result.rating
            ?.prediction ?? "",

          result.rating
            ?.confidence ?? "",

          escapeCsv(
            result.emotion?.dominant || ""
          ),

          result.emotion
            ?.confidence ?? "",

          escapeCsv(
            primaryAspect.aspect || ""
          ),

          escapeCsv(
            primaryAspect.sentiment || ""
          ),

          escapeCsv(
            result.quality?.status || ""
          )
        ].join(",");
      }
    );

    const csvContent = [
      headers.join(","),
      ...rows
    ].join("\n");

    const blob = new Blob(
      [csvContent],
      {
        type: "text/csv;charset=utf-8;"
      }
    );

    const url =
      window.URL.createObjectURL(
        blob
      );

    const link =
      document.createElement("a");

    link.href = url;

    link.setAttribute(
      "download",
      `hotel-review-batch-${Date.now()}.csv`
    );

    document.body.appendChild(
      link
    );

    link.click();

    document.body.removeChild(
      link
    );

    window.URL.revokeObjectURL(url);
  }

  function escapeCsv(value) {
    return `"${String(
      value ?? ""
    ).replace(/"/g, '""')}"`;
  }

  return (
    <div className="card">
      <h3>Download Results</h3>

      <button
        className="primary-btn"
        disabled={!results.length}
        onClick={downloadCsv}
      >
        Download CSV
      </button>

      {results.length > 0 && (
        <p className="download-note">
          Export {results.length} analyzed
          review
          {results.length !== 1
            ? "s"
            : ""}
          .
        </p>
      )}
    </div>
  );
}