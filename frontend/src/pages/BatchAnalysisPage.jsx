import { useState } from "react";

import BatchUploadPanel from "../components/batch/BatchUploadPanel";
import BatchValidationSummary from "../components/batch/BatchValidationSummary";
import BatchPreviewTable from "../components/batch/BatchPreviewTable";
import BatchResultsTable from "../components/batch/BatchResultsTable";
import BatchDownloadActions from "../components/batch/BatchDownloadActions";

import useBatchAnalysis from "../hooks/useBatchAnalysis";

import {
  parseCsv,
  validateCsv
} from "../utils/csv";

export default function BatchAnalysisPage() {
  const [fileName, setFileName] = useState("");
  const [rows, setRows] = useState([]);
  const [errors, setErrors] = useState([]);
  const [columns, setColumns] = useState([]);

  const {
    status,
    results,
    summary,
    error,
    run,
    reset: resetBatch
  } = useBatchAnalysis();

  async function handleFile(file) {
    try {
      const parsed = await parseCsv(file);

      const validation = validateCsv(parsed);

      setFileName(file.name);
      setRows(validation.rows || []);
      setErrors(validation.errors || []);
      setColumns(validation.columns || []);

      resetBatch();
    } catch (err) {
      setErrors([
        err.message ||
          "Failed to parse CSV file."
      ]);

      setRows([]);
      setColumns([]);
      resetBatch();
    }
  }

  async function analyzeBatch() {
    if (!rows.length) {
      return;
    }

    await run(rows);
  }

  function reset() {
    setFileName("");
    setRows([]);
    setErrors([]);
    setColumns([]);

    resetBatch();
  }

  return (
    <div className="analysis-page">
      {/* LEFT PANEL */}
      <div className="review-workspace">
        <BatchUploadPanel
          fileName={fileName}
          onFileSelect={handleFile}
          onReset={reset}
        />

        <BatchValidationSummary
          fileName={fileName}
          rowCount={rows.length}
          columnCount={columns.length}
          validRows={rows.length}
          errorCount={errors.length}
        />

        <div className="card">
          <h3>Batch Actions</h3>

          <button
            className="primary-btn"
            disabled={
              !rows.length ||
              status === "loading"
            }
            onClick={analyzeBatch}
          >
            {status === "loading"
              ? "Analyzing..."
              : "Analyze Batch"}
          </button>

          {status === "success" && (
            <p className="success-text">
              Analysis completed for{" "}
              {summary?.totalReviews || 0}
              {" "}reviews.
            </p>
          )}

          {error && (
            <p className="error-text">
              {error}
            </p>
          )}
        </div>

        {status === "success" &&
          results.length > 0 && (
            <BatchDownloadActions
              results={results}
            />
          )}
      </div>

      {/* RIGHT PANEL */}
      <section className="intelligence-workspace">
        {!fileName && (
          <div className="placeholder-page">
            <h2>Upload a CSV File</h2>

            <p>
              Upload a CSV containing
              hotel reviews to validate,
              preview, and analyze in
              bulk.
            </p>
          </div>
        )}

        {!!errors.length && (
          <div className="warning-list">
            {errors.map(
              (message, index) => (
                <p key={index}>
                  ⚠ {message}
                </p>
              )
            )}
          </div>
        )}

        {status === "loading" && (
          <div className="placeholder-page">
            <h2>
              Analyzing Reviews...
            </h2>

            <p>
              Running batch analysis
              through the Hotel Review
              Intelligence engine.
            </p>
          </div>
        )}

        {status === "success" && (
          <BatchResultsTable
            results={results}
          />
        )}

        {status !== "success" &&
          rows.length > 0 && (
            <BatchPreviewTable
              rows={rows}
            />
          )}
      </section>
    </div>
  );
}