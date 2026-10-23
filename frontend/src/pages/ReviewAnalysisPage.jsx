import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import ReviewWorkspace from "../components/review/ReviewWorkspace";
import EmptyState from "../components/common/EmptyState";
import Skeleton from "../components/common/Skeleton";

import SentimentCard from "../components/metrics/SentimentCard";
import RatingCard from "../components/metrics/RatingCard";
import EmotionCard from "../components/metrics/EmotionCard";
import QualityCard from "../components/metrics/QualityCard";

import AspectPanel from "../components/aspects/AspectPanel";
import EvidencePanel from "../components/aspects/EvidencePanel";

import SummaryPanel from "../components/summary/SummaryPanel";

import useReviewAnalysis from "../hooks/useReviewAnalysis";
import { saveAnalysis } from "../utils/storage";

const SELECTED_HISTORY_KEY =
  "hri-selected-analysis";

export default function ReviewAnalysisPage() {
  const location = useLocation();

  const [text, setText] = useState("");
  const [selected, setSelected] =
    useState(null);

  const {
    status,
    analysis,
    error,
    run,
    loadAnalysis,
    reset
  } = useReviewAnalysis();

  async function analyze() {
    const data = await run(text);

    if (data) {
      setSelected(
        data.aspects?.[0] || null
      );

      saveAnalysis({
        reviewText: text,
        analysis: data
      });
    }
  }

  useEffect(() => {
    const historyItem =
      location.state?.historyItem;

    if (historyItem) {
      setText(historyItem.reviewText);

      loadAnalysis(
        historyItem.analysis
      );

      setSelected(
        historyItem.analysis?.aspects?.[0] ||
        null
      );

      sessionStorage.setItem(
        SELECTED_HISTORY_KEY,
        JSON.stringify(historyItem)
      );

      return;
    }

    const stored =
      sessionStorage.getItem(
        SELECTED_HISTORY_KEY
      );

    if (!stored) return;

    try {
      const item = JSON.parse(stored);

      setText(item.reviewText);

      loadAnalysis(item.analysis);

      setSelected(
        item.analysis?.aspects?.[0] ||
        null
      );
    } catch {
      // ignore
    }
  }, [location.state, loadAnalysis]);

  useEffect(() => {
    const fn = (e) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key === "Enter" &&
        text.trim()
      ) {
        analyze();
      }
    };

    addEventListener("keydown", fn);

    return () =>
      removeEventListener("keydown", fn);
  });

  function clear() {
    setText("");
    setSelected(null);

    sessionStorage.removeItem(
      SELECTED_HISTORY_KEY
    );

    reset();
  }

  return (
    <div className="analysis-page">
      <ReviewWorkspace
        text={text}
        setText={setText}
        onAnalyze={analyze}
        onClear={clear}
        status={status}
        error={error}
      />

      <section className="intelligence-workspace">
        {status === "idle" && <EmptyState />}

        {status === "loading" && (
          <Skeleton />
        )}

        {status === "error" &&
          !analysis && <EmptyState />}

        {analysis && (
          <>
            <div className="metric-grid">
              <SentimentCard
                data={analysis.sentiment}
              />

              <RatingCard
                data={analysis.rating}
              />

              <EmotionCard
                data={analysis.emotion}
              />

              <QualityCard
                data={analysis.quality}
              />
            </div>

            {analysis.quality?.warnings
              ?.length > 0 && (
              <div className="warning-list">
                {analysis.quality.warnings.map(
                  (warning, index) => (
                    <p key={index}>
                      ⚠ {warning}
                    </p>
                  )
                )}
              </div>
            )}

            <div className="intelligence-grid">
              <AspectPanel
                aspects={analysis.aspects}
                selected={selected}
                onSelect={setSelected}
              />

              <SummaryPanel
                summary={analysis.summary}
              />
            </div>

            <EvidencePanel
              aspect={selected}
            />
          </>
        )}
      </section>
    </div>
  );
}