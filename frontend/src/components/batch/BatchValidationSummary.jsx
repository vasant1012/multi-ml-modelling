export default function BatchValidationSummary({
  fileName,
  rowCount,
  columnCount,
  validRows,
  errorCount
}) {
  return (
    <div className="card">
      <h3>Validation Summary</h3>

      <div className="summary-grid">
        <div className="summary-row">
          <span>Rows</span>
          <strong>{rowCount}</strong>
        </div>

        <div className="summary-row">
          <span>Valid Rows</span>
          <strong>{validRows}</strong>
        </div>

        <div className="summary-row">
          <span>Errors</span>
          <strong>{errorCount}</strong>
        </div>
      </div>
    </div>
  );
}