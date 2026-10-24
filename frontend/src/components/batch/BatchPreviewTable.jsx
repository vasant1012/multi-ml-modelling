export default function BatchPreviewTable({
  rows
}) {
  const previewRows = rows.slice(0, 50);

  return (
    <div className="card">
      <div className="table-header">
        <h3>Preview</h3>

        <span>
          Showing {previewRows.length} of{" "}
          {rows.length} rows
        </span>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Review</th>
              <th>Status</th>
            </tr>
          </thead>

          <tbody>
            {previewRows.map((row) => (
              <tr key={row.rowNumber}>
                <td>{row.rowNumber}</td>
                <td>{row.review_text}</td>
                <td>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}