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

import {
  saveLatestBatch
} from "../utils/storage";

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
      setFileName(file?.name || "");
      setRows([]);
      setColumns([]);
      setErrors([
        err.message ||
          "Failed to parse CSV file."
      ]);

      resetBatch();
    }
  }

  async function analyzeBatch() {
    if (!rows.length) {
      return;
    }

    const batchResults = await run(rows);

    if (
      Array.isArray(batchResults) &&
      batchResults.length > 0
    ) {
      saveLatestBatch({
        fileName,
        results: batchResults
      });
    }
  }

  function reset() {
    setFileName("");
    setRows([]);
    setErrors([]);
    setColumns([]);

    resetBatch();
  }

  const hasResults =
    status === "success" &&
    results.length > 0;

  return (
    <div className="batch-page">
      <section className="batch-control-grid">
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

        <div className="card batch-actions-card">
          <h3>Batch Actions</h3>

          <div className="batch-actions-content">
            <button
              type="button"
              className="primary-btn"
              disabled={
                !rows.length ||
                errors.length > 0 ||
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
                {summary?.totalReviews || 0}{" "}
                reviews.
              </p>
            )}

            {error && (
              <p className="error-text">
                {error}
              </p>
            )}

            {hasResults && (
              <BatchDownloadActions
                results={results}
              />
            )}
          </div>
        </div>
      </section>

      <section className="batch-results-workspace">
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

        {!fileName && (
          <div className="batch-empty-state">
            <h2>Upload a CSV File</h2>

            <p>
              Upload a CSV containing hotel
              reviews to validate, preview,
              and analyze in bulk.
            </p>
          </div>
        )}

        {fileName &&
          !rows.length &&
          !errors.length && (
            <div className="batch-empty-state">
              <h2>No Reviews Available</h2>

              <p>
                The uploaded CSV does not
                contain any review records
                ready for analysis.
              </p>
            </div>
          )}

        {status === "loading" && (
          <div className="batch-processing-state">
            <h2>Analyzing Reviews...</h2>

            <p>
              Running batch analysis through
              the Hotel Review Intelligence
              engine.
            </p>
          </div>
        )}

        {status !== "loading" &&
          status !== "success" &&
          rows.length > 0 && (
            <BatchPreviewTable
              rows={rows}
            />
          )}

        {hasResults && (
          <BatchResultsTable
            results={results}
          />
        )}

        {status === "success" &&
          !results.length && (
            <div className="batch-empty-state">
              <h2>
                No Analysis Results
              </h2>

              <p>
                Batch processing completed
                without any result records.
              </p>
            </div>
          )}
      </section>
    </div>
  );
}