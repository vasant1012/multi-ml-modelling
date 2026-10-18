import client from "./client";
export async function analyzeReview(reviewText){const {data}=await client.post("/analyze",{review_text:reviewText});return data}
export async function getHealth(){const {data}=await client.get("/health");return data}
export async function getModels(){const {data}=await client.get("/models");return data}
