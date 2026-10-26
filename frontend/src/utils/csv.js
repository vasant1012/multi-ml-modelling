import Papa from "papaparse";

const REQUIRED_COLUMN = "review_text";
const MAX_REVIEW_LENGTH = 20000;

export function parseCsv(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results),
      error: (error) => reject(error)
    });
  });
}

export function validateCsv(results) {
  const fields = results.meta?.fields || [];

  if (!fields.includes(REQUIRED_COLUMN)) {
    return {
      valid: false,
      errors: [
        `Missing required column "${REQUIRED_COLUMN}".`
      ],
      rows: []
    };
  }

  const rows = [];
  const errors = [];

  results.data.forEach((row, index) => {
    const reviewText = String(
      row.review_text || ""
    ).trim();

    if (!reviewText) {
      return;
    }

    if (reviewText.length > MAX_REVIEW_LENGTH) {
      errors.push(
        `Row ${index + 2}: exceeds ${MAX_REVIEW_LENGTH} characters`
      );
      return;
    }

    rows.push({
      rowNumber: index + 2,
      review_text: reviewText,
      status: "valid"
    });
  });

  return {
    valid: errors.length === 0,
    errors,
    rows,
    columns: fields
  };
}