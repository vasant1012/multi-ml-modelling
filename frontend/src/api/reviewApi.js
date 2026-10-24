import client from "./client";

/* Single Review */

export async function analyzeReview(reviewText) {
  const { data } = await client.post(
    "/analyze",
    {
      review_text: reviewText
    }
  );

  return data;
}

/* Batch Review */

export async function analyzeBatch(
  reviews
) {
  const payload = {
    reviews: reviews.map((text) => ({
      review_text: text
    }))
  };

  const { data } = await client.post(
    "/analyze-batch",
    payload
  );

  return data;
}

/* Health */

export async function getHealth() {
  const { data } =
    await client.get("/health");

  return data;
}

/* Models */

export async function getModels() {
  const { data } =
    await client.get("/models");

  return data;
}