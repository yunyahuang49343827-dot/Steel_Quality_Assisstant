import axios from "axios";

import type {
  CopilotResponse,
  DefectDistributionResponse,
  DefectIntelligenceResponse,
  DemoSampleResponse,
  ExplanationResponse,
  GlobalShapResponse,
  ModelPerformanceResponse,
  OverviewDashboardData,
  PredictionFeatures,
  PredictionResponse,
  QualityOverview,
} from "../types/dashboard";


// =========================================================
// API configuration
// =========================================================

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "/api";


// =========================================================
// Axios client
// =========================================================

const api = axios.create({
  baseURL: API_BASE_URL,

  timeout: 180000,

  headers: {
    "Content-Type":
      "application/json",
  },
});


// =========================================================
// Overview
// =========================================================

export async function getQualityOverview() {

  const response =
    await api.get<QualityOverview>(
      "/quality/overview"
    );

  return response.data;
}


export async function getDefectDistribution() {

  const response =
    await api.get<DefectDistributionResponse>(
      "/quality/distribution"
    );

  return response.data;
}


export async function getModelPerformance() {

  const response =
    await api.get<ModelPerformanceResponse>(
      "/model/performance"
    );

  return response.data;
}


export async function getGlobalDrivers(
  topN = 6
) {

  const response =
    await api.get<GlobalShapResponse>(
      "/explain/global",
      {
        params: {
          top_n: topN,
        },
      }
    );

  return response.data;
}


// =========================================================
// Defect Intelligence
// =========================================================

export async function getDefectIntelligence(
  defectType: string,
  topN = 5
) {

  const response =
    await api.get<DefectIntelligenceResponse>(
      `/explain/defect/${encodeURIComponent(
        defectType
      )}`,
      {
        params: {
          top_n: topN,
        },
      }
    );

  return response.data;
}


// =========================================================
// Prediction Lab
// =========================================================

export async function getDemoSample() {

  const response =
    await api.get<DemoSampleResponse>(
      "/demo/sample"
    );

  return response.data;
}


export async function predictDefect(
  features: PredictionFeatures
) {

  const response =
    await api.post<PredictionResponse>(
      "/predict",
      {
        features,
      }
    );

  return response.data;
}


export async function explainPrediction(
  features: PredictionFeatures,
  topN = 5
) {

  const response =
    await api.post<ExplanationResponse>(
      "/explain",
      {
        features,
        top_n: topN,
      }
    );

  return response.data;
}


// =========================================================
// AI Copilot
// =========================================================

export async function askCopilot(
  question: string
) {

  const response =
    await api.post<CopilotResponse>(
      "/copilot/chat",
      {
        question,
      }
    );

  return response.data;
}


// =========================================================
// Dashboard loader
// =========================================================

export async function getOverviewDashboardData():
  Promise<OverviewDashboardData> {

  const [
    overview,
    distributionResponse,
    performance,
    globalResponse,
  ] = await Promise.all([
    getQualityOverview(),
    getDefectDistribution(),
    getModelPerformance(),
    getGlobalDrivers(6),
  ]);

  return {
    overview,

    distribution:
      distributionResponse.distribution,

    performance,

    globalDrivers:
      globalResponse.drivers,
  };
}