// =========================================================
// Quality overview
// =========================================================

export interface QualityOverview {
  modeling_samples: number;
  defect_classes: number;
  champion_model: string;
  test_macro_f1: number;
  test_macro_f1_percentage: number;
  feature_count: number;
}


// =========================================================
// Defect distribution
// =========================================================

export interface DefectDistributionItem {
  defect_type: string;
  sample_count: number;
  percentage: number;
}

export interface DefectDistributionResponse {
  distribution: DefectDistributionItem[];
}


// =========================================================
// Model performance
// =========================================================

export interface ModelSummary {
  champion_model: string;
  selection_metric: string;
  test_accuracy: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_macro_f1: number;
  test_weighted_f1: number;
}

export interface PerClassMetric {
  defect_type: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelPerformanceResponse {
  summary: ModelSummary;
  per_class: PerClassMetric[];
  interpretation_note: string;
}


// =========================================================
// Global SHAP
// =========================================================

export interface GlobalDriver {
  rank: number;
  feature: string;
  mean_abs_shap: number;
}

export interface GlobalShapResponse {
  drivers: GlobalDriver[];
  interpretation_note: string;
}


// =========================================================
// Per-defect intelligence
// =========================================================

export interface DefectDriver {
  class: string;
  rank: number;
  feature: string;
  mean_abs_shap: number;
}

export interface DefectIntelligenceResponse {
  defect_type: string;
  samples: number;
  dataset_share: number;

  test_precision: number;
  test_recall: number;
  test_f1: number;

  drivers: DefectDriver[];
  interpretation_note: string;
}


// =========================================================
// Prediction Lab
// =========================================================

export type PredictionFeatures =
  Record<string, number>;


export interface DemoSampleResponse {
  sample_id: number;
  features: PredictionFeatures;
  note: string;
}


export interface PredictionResponse {
  predicted_defect: string;
  confidence: number;

  probabilities: Record<
    string,
    number
  >;
}


export interface LocalShapDriver {
  feature: string;
  feature_value: number;
  shap_value: number;

  direction:
    | "supports_prediction"
    | "opposes_prediction";
}


export interface ExplanationResponse {
  predicted_defect: string;
  confidence: number;

  top_drivers:
    LocalShapDriver[];

  interpretation_note: string;
}


// =========================================================
// AI Copilot
// =========================================================

export interface CopilotToolTraceItem {
  tool: string;
  arguments?: Record<
    string,
    unknown
  >;
  status: string;
  error?: string;
}


export interface CopilotResponse {
  answer: string;
  model: string | null;

  tools_used: string[];

  tool_trace:
    CopilotToolTraceItem[];

  policy_decision: string;
}


export interface ChatMessage {
  id: string;

  role:
    | "user"
    | "assistant";

  content: string;

  model?: string | null;

  toolsUsed?: string[];

  policyDecision?: string;
}


// =========================================================
// Combined overview data
// =========================================================

export interface OverviewDashboardData {
  overview: QualityOverview;
  distribution: DefectDistributionItem[];
  performance: ModelPerformanceResponse;
  globalDrivers: GlobalDriver[];
}  