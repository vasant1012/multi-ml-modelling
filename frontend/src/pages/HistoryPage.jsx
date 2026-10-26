import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { clearHistory, getHistory } from "../utils/storage";
import Button from "../components/common/Button";
import { titleCase } from "../utils/formatters";

const SELECTED_HISTORY_KEY = "hri-selected-analysis";

export default function HistoryPage() {
  const navigate = useNavigate();
  const [rows, setRows] = useState(getHistory());

  function openAnalysis(row) {
    sessionStorage.setItem(
      SELECTED_HISTORY_KEY,
      JSON.stringify(row)
    );

    navigate("/analyze", {
      state: {
        historyItem: row
      }
    });
  }

  return (
    <div className="history-page">
      <div className="page-header">
        <div>
          <span className="eyebrow">Recent activity</span>
          <h2>Analysis history</h2>
          <p>
            The most recent 20 analyses stored in this browser.
          </p>
        </div>

        <Button
          variant="secondary"
          onClick={() => {
            clearHistory();
            setRows([]);
          }}
          disabled={!rows.length}
        >
          Clear history
        </Button>
      </div>

      {rows.length ? (
        <div className="history-list">
          {rows.map((row) => (
            <article
              key={row.id}
              role="button"
              tabIndex={0}
              className="history-item clickable"
              onClick={() => openAnalysis(row)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" ||
                  e.key === " "
                ) {
                  e.preventDefault();
                  openAnalysis(row);
                }
              }}
            >
              <div>
                <strong>
                  {row.reviewText.slice(0, 110)}
                  {row.reviewText.length > 110
                    ? "…"
                    : ""}
                </strong>

                <small>
                  {new Date(
                    row.createdAt
                  ).toLocaleString()}
                </small>
              </div>

              <span>
                {titleCase(
                  row.analysis?.sentiment?.label
                )}{" "}
                ·{" "}
                {row.analysis?.rating?.prediction}/5
              </span>
            </article>
          ))}
        </div>
      ) : (
        <div className="placeholder-page">
          <h2>No analysis history</h2>
          <p>
            Completed review analyses will appear
            here.
          </p>
        </div>
      )}
    </div>
  );
}