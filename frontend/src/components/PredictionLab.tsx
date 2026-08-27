import {
  useMemo,
  useState,
} from "react";

import {
  AlertTriangle,
  Beaker,
  BrainCircuit,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  Info,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";

import {
  explainPrediction,
  getDemoSample,
  predictDefect,
} from "../api/dashboard";

import type {
  ExplanationResponse,
  PredictionFeatures,
  PredictionResponse,
} from "../types/dashboard";


// =========================================================
// Feature groups
// =========================================================

const FEATURE_GROUPS = {
  "生產 / 鋼材": [
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
  ],

  "幾何與位置": [
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
  ],

  "亮度特徵": [
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Luminosity_Index",
  ],

  "形狀 / 邊緣": [
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "SigmoidOfAreas",
  ],
};


// =========================================================
// Helpers
// =========================================================

function formatPercent(
  value: number
) {
  return `${(
    value * 100
  ).toFixed(2)}%`;
}


function humanizeFeature(
  feature: string
) {
  return feature.replaceAll(
    "_",
    " "
  );
}


// =========================================================
// Feature accordion
// =========================================================

function FeatureGroup({
  title,
  fields,
  values,
  onChange,
  defaultOpen = false,
}: {
  title: string;
  fields: string[];
  values: PredictionFeatures;
  onChange: (
    field: string,
    value: number
  ) => void;
  defaultOpen?: boolean;
}) {

  const [
    open,
    setOpen,
  ] = useState(
    defaultOpen
  );

  return (
    <div className="feature-group">

      <button
        className="feature-group-header"
        onClick={
          () =>
            setOpen(
              !open
            )
        }
      >

        <span>
          {title}
        </span>

        {open
          ? <ChevronUp size={16} />
          : <ChevronDown size={16} />
        }

      </button>

      {open && (

        <div className="feature-input-grid">

          {fields.map(
            field => (
              <label
                className="feature-input"
                key={field}
              >

                <span>
                  {humanizeFeature(
                    field
                  )}
                </span>

                <input
                  type="number"
                  step="any"
                  value={
                    values[field]
                    ?? ""
                  }
                  onChange={
                    event =>
                      onChange(
                        field,
                        Number(
                          event.target.value
                        )
                      )
                  }
                />

              </label>
            )
          )}

        </div>
      )}

    </div>
  );
}


// =========================================================
// Prediction probabilities
// =========================================================

function ProbabilityPanel({
  prediction,
}: {
  prediction: PredictionResponse;
}) {

  const probabilities =
    Object.entries(
      prediction.probabilities
    )
    .sort(
      (
        a,
        b
      ) =>
        b[1] - a[1]
    );

  return (
    <div className="prediction-result-card">

      <div className="prediction-result-header">

        <span>
          預測結果
        </span>

        <div className="prediction-status">
          模型推論
        </div>

      </div>

      <div className="predicted-class">

        <span>
          預測缺陷
        </span>

        <h3>
          {
            prediction
              .predicted_defect
          }
        </h3>

      </div>

      <div className="confidence-block">

        <span>
          模型 Confidence
        </span>

        <strong>
          {formatPercent(
            prediction.confidence
          )}
        </strong>

      </div>


      <div className="probability-list">

        {probabilities.map(
          ([
            defect,
            probability,
          ]) => (

            <div
              className="probability-row"
              key={defect}
            >

              <div className="probability-label">

                <span>
                  {defect}
                </span>

                <strong>
                  {formatPercent(
                    probability
                  )}
                </strong>

              </div>

              <div className="probability-track">

                <div
                  className="probability-fill"
                  style={{
                    width:
                      `${probability * 100}%`,
                  }}
                />

              </div>

            </div>
          )
        )}

      </div>


      {prediction.confidence < 0.6 && (

        <div className="uncertainty-note">

          <AlertTriangle
            size={15}
          />

          <span>
            此筆預測的模型 Confidence 較低，
            建議僅作為工程判斷輔助，
            不應直接作為自動品質處置依據。
          </span>

        </div>
      )}

    </div>
  );
}


// =========================================================
// Local SHAP
// =========================================================

function ExplanationPanel({
  explanation,
}: {
  explanation: ExplanationResponse;
}) {

  const maxAbs =
    Math.max(
      ...explanation
        .top_drivers
        .map(
          driver =>
            Math.abs(
              driver.shap_value
            )
        ),
      0.0001
    );

  return (
    <div className="local-shap-card">

      <div className="local-shap-header">

        <div>

          <span>
            SHAP 可解釋性
          </span>

          <h3>
            為什麼模型預測為{" "}
            {
              explanation
                .predicted_defect
            }？
          </h3>

        </div>

        <BrainCircuit
          size={22}
        />

      </div>


      <div className="local-shap-list">

        {explanation
          .top_drivers
          .map(
            driver => {

              const positive =
                driver.shap_value > 0;

              const width =
                Math.abs(
                  driver.shap_value
                )
                / maxAbs
                * 100;

              return (
                <div
                  className="local-shap-row"
                  key={
                    driver.feature
                  }
                >

                  <div className="local-shap-top">

                    <div className="local-shap-feature">

                      <span
                        className={
                          positive
                            ? "shap-direction positive"
                            : "shap-direction negative"
                        }
                      >
                        {positive
                          ? "▲"
                          : "▼"
                        }
                      </span>

                      <strong>
                        {
                          driver.feature
                        }
                      </strong>

                    </div>

                    <span
                      className={
                        positive
                          ? "shap-value positive"
                          : "shap-value negative"
                      }
                    >
                      {positive
                        ? "+"
                        : ""
                      }
                      {
                        driver
                          .shap_value
                          .toFixed(
                            3
                          )
                      }
                    </span>

                  </div>


                  <div className="local-shap-track">

                    <div
                      className={
                        positive
                          ? "local-shap-fill positive"
                          : "local-shap-fill negative"
                      }
                      style={{
                        width:
                          `${width}%`,
                      }}
                    />

                  </div>


                  <div className="feature-value-line">

                    特徵值：{" "}

                    <strong>
                      {
                        driver
                          .feature_value
                      }
                    </strong>

                    <span>
                      {
                        positive
                          ? "支持此預測"
                          : "降低此預測"
                      }
                    </span>

                  </div>

                </div>
              );
            }
          )}

      </div>


      <div className="causality-note">

        <ShieldCheck
          size={16}
        />

        <span>
          SHAP 用於解釋模型預測行為，
          不代表已確認的製程因果關係或製造根因。
        </span>

      </div>

    </div>
  );
}


// =========================================================
// Main Prediction Lab
// =========================================================

export default function PredictionLab() {

  const [
    features,
    setFeatures,
  ] = useState<
    PredictionFeatures
  >({});

  const [
    sampleId,
    setSampleId,
  ] = useState<
    number | null
  >(null);

  const [
    prediction,
    setPrediction,
  ] = useState<
    PredictionResponse | null
  >(null);

  const [
    explanation,
    setExplanation,
  ] = useState<
    ExplanationResponse | null
  >(null);

  const [
    loadingSample,
    setLoadingSample,
  ] = useState(
    false
  );

  const [
    running,
    setRunning,
  ] = useState(
    false
  );

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(null);


  const requiredFields =
    useMemo(
      () =>
        Object.values(
          FEATURE_GROUPS
        ).flat(),
      []
    );


  const featureCount =
    Object.keys(
      features
    ).length;


  const ready =
    requiredFields.every(
      field =>
        typeof features[field]
        === "number"
        &&
        Number.isFinite(
          features[field]
        )
    );


  function updateFeature(
    field: string,
    value: number
  ) {

    setFeatures(
      previous => ({
        ...previous,
        [field]: value,
      })
    );

    setPrediction(
      null
    );

    setExplanation(
      null
    );
  }


  async function loadDemoSample() {

    try {

      setLoadingSample(
        true
      );

      setError(
        null
      );

      const sample =
        await getDemoSample();

      setFeatures(
        sample.features
      );

      setSampleId(
        sample.sample_id
      );

      setPrediction(
        null
      );

      setExplanation(
        null
      );

    } catch (err) {

      console.error(
        err
      );

      setError(
        "無法載入 Demo 樣本，請確認 FastAPI 服務是否正常。"
      );

    } finally {

      setLoadingSample(
        false
      );
    }
  }


  async function runPrediction() {

    if (!ready) {

      setError(
        "請先載入 Demo 樣本，或完整輸入 27 個模型特徵後再執行預測。"
      );

      return;
    }

    try {

      setRunning(
        true
      );

      setError(
        null
      );

      const [
        predictionResult,
        explanationResult,
      ] = await Promise.all([
        predictDefect(
          features
        ),

        explainPrediction(
          features,
          5
        ),
      ]);

      setPrediction(
        predictionResult
      );

      setExplanation(
        explanationResult
      );

    } catch (err) {

      console.error(
        err
      );

      setError(
        "模型預測失敗，請確認 FastAPI 服務與輸入特徵是否正常。"
      );

    } finally {

      setRunning(
        false
      );
    }
  }


  function clearLab() {

    setFeatures(
      {}
    );

    setSampleId(
      null
    );

    setPrediction(
      null
    );

    setExplanation(
      null
    );

    setError(
      null
    );
  }


  return (
    <section
      className="prediction-lab-section"
      id="prediction-lab"
    >

      <div className="prediction-lab-title">

        <div>

          <span className="section-eyebrow">
            Prediction Lab
          </span>

          <h2>
            模型預測與可解釋性工作區
          </h2>

          <p>
            使用最佳 XGBoost 模型進行缺陷預測，
            查看各類別預測機率，
            並透過 SHAP 解釋單筆預測結果。
          </p>

        </div>

        <div className="prediction-lab-badge">

          <Beaker size={15} />

          互動式 ML 工作區

        </div>

      </div>


      <div className="prediction-lab-grid">

        {/* ============================================= */}
        {/* INPUT */}
        {/* ============================================= */}

        <div className="sample-input-card">

          <div className="sample-input-header">

            <div>

              <span>
                模型輸入
              </span>

              <h3>
                鋼材品質特徵
              </h3>

            </div>

            <div className="feature-counter">

              {featureCount}
              /27

            </div>

          </div>


          {sampleId !== null && (

            <div className="demo-sample-banner">

              <FlaskConical
                size={15}
              />

              保留 Demo 樣本

              <strong>
                ID {sampleId}
              </strong>

            </div>
          )}


          <div className="feature-groups">

            {Object.entries(
              FEATURE_GROUPS
            ).map(
              ([
                title,
                fields,
              ], index) => (

                <FeatureGroup
                  key={title}
                  title={title}
                  fields={fields}
                  values={features}
                  onChange={
                    updateFeature
                  }
                  defaultOpen={
                    index === 0
                  }
                />

              )
            )}

          </div>


          {error && (

            <div className="prediction-error">

              <AlertTriangle
                size={15}
              />

              {error}

            </div>
          )}


          <div className="prediction-actions">

            <button
              className="secondary-action"
              onClick={
                loadDemoSample
              }
              disabled={
                loadingSample
              }
            >

              {loadingSample
                ? (
                  <LoaderCircle
                    className="spin"
                    size={16}
                  />
                )
                : (
                  <FlaskConical
                    size={16}
                  />
                )
              }

              載入 Demo 樣本

            </button>


            <button
              className="primary-action"
              onClick={
                runPrediction
              }
              disabled={
                running
              }
            >

              {running
                ? (
                  <LoaderCircle
                    className="spin"
                    size={16}
                  />
                )
                : (
                  <Play
                    size={16}
                  />
                )
              }

              執行預測

            </button>


            <button
              className="clear-action"
              onClick={
                clearLab
              }
              title="重設 Prediction Lab"
            >

              <RotateCcw
                size={16}
              />

            </button>

          </div>

        </div>


        {/* ============================================= */}
        {/* PREDICTION */}
        {/* ============================================= */}

        {prediction
          ? (
            <ProbabilityPanel
              prediction={
                prediction
              }
            />
          )
          : (
            <div className="prediction-empty-card">

              <Beaker
                size={31}
              />

              <h3>
                尚未執行預測
              </h3>

              <p>
                請載入保留的 Demo 樣本，
                或完整輸入 27 個模型特徵後執行預測。
              </p>

            </div>
          )
        }


        {/* ============================================= */}
        {/* EXPLANATION */}
        {/* ============================================= */}

        {explanation
          ? (
            <ExplanationPanel
              explanation={
                explanation
              }
            />
          )
          : (
            <div className="prediction-empty-card">

              <BrainCircuit
                size={31}
              />

              <h3>
                SHAP 單筆預測解釋
              </h3>

              <p>
                執行模型預測後，
                此處將顯示影響該筆預測的主要特徵。
              </p>

              <div className="empty-guardrail">

                <Info size={14} />

                SHAP 解釋模型行為，
                不代表製程因果關係。

              </div>

            </div>
          )
        }

      </div>

    </section>
  );
}