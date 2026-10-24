const HISTORY_KEY =
  "hri-history";

const LATEST_BATCH_KEY =
  "hri-latest-batch";

function readJson(key, fallback) {
  try {
    const storedValue =
      localStorage.getItem(key);

    if (!storedValue) {
      return fallback;
    }

    return JSON.parse(storedValue);
  } catch {
    return fallback;
  }
}

function createId() {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID ===
      "function"
  ) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random()
    .toString(16)
    .slice(2)}`;
}

export function getHistory() {
  const history = readJson(
    HISTORY_KEY,
    []
  );

  return Array.isArray(history)
    ? history
    : [];
}

export function saveAnalysis(item) {
  const historyItem = {
    id: createId(),
    createdAt:
      new Date().toISOString(),
    ...item
  };

  const rows = [
    historyItem,
    ...getHistory()
  ].slice(0, 20);

  localStorage.setItem(
    HISTORY_KEY,
    JSON.stringify(rows)
  );

  return rows;
}

export function clearHistory() {
  localStorage.removeItem(
    HISTORY_KEY
  );
}

export function getLatestBatch() {
  const batch = readJson(
    LATEST_BATCH_KEY,
    null
  );

  if (
    !batch ||
    typeof batch !== "object" ||
    !Array.isArray(batch.results)
  ) {
    return null;
  }

  return batch;
}

export function saveLatestBatch({
  fileName,
  results
}) {
  if (
    !Array.isArray(results) ||
    results.length === 0
  ) {
    return null;
  }

  const batch = {
    id: createId(),
    createdAt:
      new Date().toISOString(),
    fileName:
      fileName ||
      "Uploaded CSV",
    results
  };

  localStorage.setItem(
    LATEST_BATCH_KEY,
    JSON.stringify(batch)
  );

  return batch;
}

export function clearLatestBatch() {
  localStorage.removeItem(
    LATEST_BATCH_KEY
  );
}