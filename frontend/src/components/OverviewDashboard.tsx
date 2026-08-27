import {
  useEffect,
  useState,
} from "react";

import type {
  ReactNode,
} from "react";

import {
  AlertTriangle,
  Boxes,
  Database,
  Gauge,
  Info,
  LoaderCircle,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  Trophy,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getDefectIntelligence,
} from "../api/dashboard";

import type {
  DefectIntelligenceResponse,
  GlobalDriver,
  OverviewDashboardData,
  PerClassMetric,
} from "../types/dashboard";


// =========================================================
// Props
// =========================================================

interface OverviewDashboardProps {
  data: OverviewDashboardData;
}


// =========================================================
// Formatting helpers
// =========================================================

function formatNumber(
  value: number
) {
  return value.toLocaleString(
    "en-US"
  );
}


function formatPercent(
  value: number
) {
  return `${(
    value * 100
  ).toFixed(2)}%`;
}


// =========================================================
// KPI Card
// =========================================================

interface KpiCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: ReactNode;
  accent?: "blue" | "teal";
}


function KpiCard({
  title,
  value,
  subtitle,
  icon,
  accent = "blue",
}: KpiCardProps) {

  return (
    <div className="kpi-card">

      <div
        className={
          `kpi-icon ${accent}`
        }
      >
        {icon}
      </div>

      <div className="kpi-content">

        <span className="kpi-title">
          {title}
        </span>

        <strong className="kpi-value">
          {value}
        </strong>

        <span className="kpi-subtitle">
          {subtitle}
        </span>

      </div>

    </div>
  );
}


// =========================================================
// Panel header
// =========================================================

function PanelHeader({
  title,
  subtitle,
  badge,
}: {
  title: string;
  subtitle?: string;
  badge?: string;
}) {

  return (
    <div className="panel-header">

      <div>

        <h2>
          {title}
        </h2>

        {subtitle && (
          <p>
            {subtitle}
          </p>
        )}

      </div>

      {badge && (
        <span className="evidence-badge">
          {badge}
        </span>
      )}

    </div>
  );
}


// =========================================================
// Distribution tooltip
// =========================================================

function DistributionTooltip({
  active,
  payload,
}: any) {

  if (
    !active ||
    !payload?.length
  ) {
    return null;
  }

  const item =
    payload[0].payload;

  return (
    <div className="chart-tooltip">

      <strong>
        {item.defect_type}
      </strong>

      <span>
        {formatNumber(
          item.sample_count
        )} 筆樣本
      </span>

      <span>
        占比{" "}
        {item.percentage.toFixed(
          2
        )}%
      </span>

    </div>
  );
}


// =========================================================
// Defect distribution
// =========================================================

function DefectDistributionPanel({
  data,
}: {
  data: OverviewDashboardData[
    "distribution"
  ];
}) {

  return (
    <section className="dashboard-panel">

      <PanelHeader
        title="缺陷數量分布"
        subtitle="建模資料集中各缺陷類別的樣本數量"
      />

      <div className="chart-large">

        <ResponsiveContainer
          width="100%"
          height="100%"
        >

          <BarChart
            data={data}
            layout="vertical"
            margin={{
              top: 4,
              right: 45,
              bottom: 4,
              left: 10,
            }}
          >

            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="#edf1f7"
            />

            <XAxis
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{
                fill: "#8190a5",
                fontSize: 11,
              }}
            />

            <YAxis
              dataKey="defect_type"
              type="category"
              width={95}
              axisLine={false}
              tickLine={false}
              tick={{
                fill: "#44536a",
                fontSize: 11,
              }}
            />

            <Tooltip
              cursor={{
                fill:
                  "rgba(37, 99, 235, 0.04)",
              }}
              content={
                <DistributionTooltip />
              }
            />

            <Bar
              dataKey="sample_count"
              radius={[
                0,
                5,
                5,
                0,
              ]}
              fill="#2468d8"
              maxBarSize={20}
            />

          </BarChart>

        </ResponsiveContainer>

      </div>

      <div className="distribution-foot">

        {data
          .slice(0, 3)
          .map(
            item => (
              <span
                key={
                  item.defect_type
                }
              >
                <strong>
                  {
                    item
                      .defect_type
                  }
                </strong>

                {" "}

                {item.percentage.toFixed(
                  2
                )}%
              </span>
            )
          )}

      </div>

    </section>
  );
}


// =========================================================
// Composition
// =========================================================

const PIE_COLORS = [
  "#225fd1",
  "#2777dd",
  "#2d91dc",
  "#43a9da",
  "#63bdd9",
  "#83cddc",
  "#afdfe7",
];


function DefectCompositionPanel({
  data,
}: {
  data: OverviewDashboardData[
    "distribution"
  ];
}) {

  const total =
    data.reduce(
      (
        sum,
        item
      ) =>
        sum
        + item.sample_count,
      0
    );

  return (
    <section className="dashboard-panel">

      <PanelHeader
        title="缺陷組成"
        subtitle="各缺陷類別占建模資料集的比例"
      />

      <div className="composition-layout">

        <div className="donut-wrapper">

          <ResponsiveContainer
            width="100%"
            height="100%"
          >

            <PieChart>

              <Pie
                data={data}
                dataKey="sample_count"
                nameKey="defect_type"
                cx="50%"
                cy="50%"
                innerRadius={65}
                outerRadius={92}
                paddingAngle={1.5}
                stroke="#ffffff"
                strokeWidth={2}
              >

                {data.map(
                  (
                    item,
                    index
                  ) => (
                    <Cell
                      key={
                        item.defect_type
                      }
                      fill={
                        PIE_COLORS[
                          index
                          % PIE_COLORS.length
                        ]
                      }
                    />
                  )
                )}

              </Pie>

              <Tooltip
                content={
                  <DistributionTooltip />
                }
              />

            </PieChart>

          </ResponsiveContainer>


          <div className="donut-center">

            <strong>
              {formatNumber(
                total
              )}
            </strong>

            <span>
              筆樣本
            </span>

          </div>

        </div>


        <div className="legend-list">

          {data.map(
            (
              item,
              index
            ) => (
              <div
                className="legend-row"
                key={
                  item.defect_type
                }
              >

                <span
                  className="legend-dot"
                  style={{
                    background:
                      PIE_COLORS[
                        index
                        % PIE_COLORS.length
                      ],
                  }}
                />

                <span className="legend-name">
                  {
                    item.defect_type
                  }
                </span>

                <strong>
                  {item.percentage.toFixed(
                    2
                  )}%
                </strong>

              </div>
            )
          )}

        </div>

      </div>

    </section>
  );
}
   
      
   
     
 
 



// =========================================================
// Performance
// =========================================================

function ModelPerformancePanel({
  data,
}: {
  data: PerClassMetric[];
}) {

  const sorted = [
    ...data,
  ].sort(
    (
      a,
      b
    ) =>
      b.recall
      - a.recall
  );

  const chartData =
    sorted.map(
      item => ({
        ...item,
        recallPercent:
          item.recall * 100,
      })
    );

  return (
    <section className="dashboard-panel">

      <PanelHeader
        title="各缺陷模型表現"
        subtitle="以各類別 Recall 比較模型對不同缺陷的辨識表現"
      />

      <div className="chart-large">

        <ResponsiveContainer
          width="100%"
          height="100%"
        >

          <BarChart
            data={chartData}
            layout="vertical"
            margin={{
              top: 4,
              right: 45,
              bottom: 4,
              left: 10,
            }}
          >

            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="#edf1f7"
            />

            <XAxis
              type="number"
              domain={[
                0,
                100,
              ]}
              tickFormatter={
                value =>
                  `${value}%`
              }
              axisLine={false}
              tickLine={false}
              tick={{
                fill: "#8190a5",
                fontSize: 11,
              }}
            />

            <YAxis
              type="category"
              dataKey="defect_type"
              width={95}
              axisLine={false}
              tickLine={false}
              tick={{
                fill: "#44536a",
                fontSize: 11,
              }}
            />

            <Tooltip
              formatter={(
                value: any
              ) => [
                `${Number(
                  value
                ).toFixed(2)}%`,
                "Recall",
              ]}
            />

            <Bar
              dataKey="recallPercent"
              fill="#1e6add"
              radius={[
                0,
                5,
                5,
                0,
              ]}
              maxBarSize={16}
            />

          </BarChart>

        </ResponsiveContainer>

      </div>

      <div className="performance-note">

        <Info size={15} />

        <span>
          整體 Accuracy 無法完整反映少數缺陷類別的辨識表現。
        </span>

      </div>

    </section>
  );
}


// =========================================================
// Global drivers
// =========================================================

function DriverImpactLabel({
  index,
}: {
  index: number;
}) {

  if (
    index <= 1
  ) {

    return (
      <span className="impact high">
        高影響
      </span>
    );
  }

  if (
    index <= 3
  ) {

    return (
      <span className="impact medium">
        中等
      </span>
    );
  }

  return (
    <span className="impact low">
      較低
    </span>
  );
}


function GlobalDriversPanel({
  data,
}: {
  data: GlobalDriver[];
}) {

  const maxValue =
    Math.max(
      ...data.map(
        item =>
          item.mean_abs_shap
      )
    );

  return (
    <section className="dashboard-panel">

      <PanelHeader
        title="全域模型預測因子"
        subtitle="各特徵在模型預測中的平均 SHAP 影響程度"
        badge="預測因子 ≠ 已確認製程根因"
      />

      <div className="drivers-list">

        {data.map(
          (
            item,
            index
          ) => {

            const width =
              maxValue > 0
                ? (
                    item.mean_abs_shap
                    / maxValue
                  ) * 100
                : 0;

            return (
              <div
                className="driver-row"
                key={
                  item.feature
                }
              >

                <div className="driver-rank">
                  {String(
                    item.rank
                  ).padStart(
                    2,
                    "0"
                  )}
                </div>

                <div className="driver-main">

                  <div className="driver-topline">

                    <span className="driver-name">
                      {item.feature}
                    </span>

                    <span className="driver-value">
                      Mean |SHAP|{" "}
                      {
                        item
                          .mean_abs_shap
                          .toFixed(3)
                      }
                    </span>

                  </div>

                  <div className="driver-track">

                    <div
                      className="driver-bar"
                      style={{
                        width:
                          `${width}%`,
                      }}
                    />

                  </div>

                </div>

                <DriverImpactLabel
                  index={index}
                />

              </div>
            );
          }
        )}

      </div>

    </section>
  );
}


// =========================================================
// Defect Intelligence
// =========================================================

const DEFECT_TYPES = [
  "Bumps",
  "Dirtiness",
  "K_Scatch",
  "Other_Faults",
  "Pastry",
  "Stains",
  "Z_Scratch",
];


function MetricTile({
  label,
  value,
  icon,
  accent = "blue",
}: {
  label: string;
  value: string;
  icon: ReactNode;
  accent?: "blue" | "teal" | "amber";
}) {

  return (
    <div className="defect-metric-tile">

      <div
        className={
          `defect-metric-icon ${accent}`
        }
      >
        {icon}
      </div>

      <div>

        <span>
          {label}
        </span>

        <strong>
          {value}
        </strong>

      </div>

    </div>
  );
}


function ScoreRing({
  label,
  value,
}: {
  label: string;
  value: number;
}) {

  const percentage =
    Math.max(
      0,
      Math.min(
        100,
        value * 100
      )
    );

  return (
    <div className="score-ring-card">

      <div
        className="score-ring"
        style={{
          background:
            `conic-gradient(
              #0b9e9a 0 ${percentage}%,
              #e9eff5 ${percentage}% 100%
            )`,
        }}
      >

        <div className="score-ring-inner">

          <strong>
            {percentage.toFixed(
              1
            )}%
          </strong>

        </div>

      </div>

      <span>
        {label}
      </span>

    </div>
  );
}


function DefectIntelligenceSection() {

  const [
    selectedDefect,
    setSelectedDefect,
  ] = useState(
    "K_Scatch"
  );

  const [
    defectData,
    setDefectData,
  ] = useState<
    DefectIntelligenceResponse | null
  >(null);

  const [
    loading,
    setLoading,
  ] = useState(
    true
  );

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(null);


  useEffect(
    () => {

      async function loadDefect() {

        try {

          setLoading(
            true
          );

          setError(
            null
          );

          const result =
            await getDefectIntelligence(
              selectedDefect,
              5
            );

          setDefectData(
            result
          );

        } catch (err) {

          console.error(
            err
          );

          setError(
            "無法載入缺陷分析資料，請確認 FastAPI 服務是否正常。"
          );

        } finally {

          setLoading(
            false
          );
        }
      }

      loadDefect();

    },
    [
      selectedDefect
    ]
  );


  return (
    <section
      className="defect-intelligence-section"
      id="defect-intelligence"
    >

      <div className="defect-section-header">

        <div>

          <span className="section-eyebrow">
            缺陷分析
          </span>

          <h2>
            探索各缺陷類別
          </h2>

          <p>
            比較缺陷占比、Holdout Test
            表現與 SHAP 預測證據。
          </p>

        </div>

        <div className="defect-guardrail">

          <ShieldCheck
            size={15}
          />

          預測證據，不代表製程根因

        </div>

      </div>


      <div className="defect-tabs">

        {DEFECT_TYPES.map(
          defect => (
            <button
              key={defect}
              className={
                selectedDefect
                === defect
                  ? "defect-tab active"
                  : "defect-tab"
              }
              onClick={
                () =>
                  setSelectedDefect(
                    defect
                  )
              }
            >
              {defect}
            </button>
          )
        )}

      </div>


      {loading && (

        <div className="defect-loading">

          <LoaderCircle
            size={24}
            className="spin"
          />

          正在載入{" "}
          {selectedDefect}
          {" "}
          缺陷分析...

        </div>
      )}


      {error && (

        <div className="defect-error">

          <AlertTriangle
            size={20}
          />

          {error}

        </div>
      )}


      {!loading &&
        !error &&
        defectData && (

        <div className="defect-intelligence-grid">

          <div className="defect-profile-card">

            <div className="defect-profile-title">

              <div className="defect-symbol">

                <Sparkles
                  size={22}
                />

              </div>

              <div>

                <span>
                  目前缺陷類別
                </span>

                <h3>
                  {
                    defectData
                      .defect_type
                  }
                </h3>

              </div>

            </div>


            <div className="defect-metrics-grid">

              <MetricTile
                label="樣本數"
                value={
                  formatNumber(
                    defectData
                      .samples
                  )
                }
                icon={
                  <Database
                    size={17}
                  />
                }
              />

              <MetricTile
                label="資料占比"
                value={
                  `${
                    defectData
                      .dataset_share
                      .toFixed(
                        2
                      )
                  }%`
                }
                icon={
                  <Boxes
                    size={17}
                  />
                }
              />

              <MetricTile
                label="測試集 Recall"
                value={
                  formatPercent(
                    defectData
                      .test_recall
                  )
                }
                icon={
                  <Target
                    size={17}
                  />
                }
                accent="teal"
              />

              <MetricTile
                label="測試集 F1"
                value={
                  formatPercent(
                    defectData
                      .test_f1
                  )
                }
                icon={
                  <Gauge
                    size={17}
                  />
                }
                accent="teal"
              />

            </div>


            <div className="defect-score-row">

              <ScoreRing
                label="Precision"
                value={
                  defectData
                    .test_precision
                }
              />

              <ScoreRing
                label="Recall"
                value={
                  defectData
                    .test_recall
                }
              />

              <ScoreRing
                label="F1"
                value={
                  defectData
                    .test_f1
                }
              />

            </div>

          </div>


          <div className="defect-drivers-card">

            <div className="defect-drivers-header">

              <div>

                <span>
                  SHAP 可解釋性
                </span>

                <h3>
                  主要預測因子
                </h3>

              </div>

              <div className="shap-pill">
                Mean |SHAP|
              </div>

            </div>


            <div className="defect-driver-list">

              {(() => {

                const maxDriver =
                  Math.max(
                    ...defectData
                      .drivers
                      .map(
                        item =>
                          item
                            .mean_abs_shap
                      )
                  );

                return defectData
                  .drivers
                  .map(
                    driver => {

                      const width =
                        maxDriver > 0
                          ? (
                              driver
                                .mean_abs_shap
                              / maxDriver
                            ) * 100
                          : 0;

                      return (
                        <div
                          className="defect-driver-item"
                          key={
                            driver.feature
                          }
                        >

                          <div className="defect-driver-number">

                            {String(
                              driver.rank
                            ).padStart(
                              2,
                              "0"
                            )}

                          </div>

                          <div className="defect-driver-content">

                            <div className="defect-driver-label">

                              <strong>
                                {
                                  driver
                                    .feature
                                }
                              </strong>

                              <span>
                                {
                                  driver
                                    .mean_abs_shap
                                    .toFixed(
                                      3
                                    )
                                }
                              </span>

                            </div>

                            <div className="defect-driver-track">

                              <div
                                className="defect-driver-fill"
                                style={{
                                  width:
                                    `${width}%`,
                                }}
                              />

                            </div>

                          </div>

                        </div>
                      );
                    }
                  );

              })()}

            </div>


            <div className="shap-interpretation">

              <Info
                size={15}
              />

              <span>
                {
                  defectData
                    .interpretation_note
                }
              </span>

            </div>

          </div>

        </div>
      )}

    </section>
  );
}


// =========================================================
// Main Dashboard
// =========================================================

export default function OverviewDashboard({
  data,
}: OverviewDashboardProps) {

  return (
    <div>

      <section className="kpi-grid">

        <KpiCard
          title="建模樣本數"
          value={
            formatNumber(
              data
                .overview
                .modeling_samples
            )
          }
          subtitle={
            `${data.overview.feature_count} 個模型特徵`
          }
          icon={
            <Database size={22} />
          }
        />

        <KpiCard
          title="缺陷類別"
          value={
            String(
              data
                .overview
                .defect_classes
            )
          }
          subtitle="單標籤分類"
          icon={
            <Boxes size={22} />
          }
        />

        <KpiCard
          title="最佳模型"
          value={
            data
              .overview
              .champion_model
          }
          subtitle="依 Validation Macro F1 選定"
          icon={
            <Trophy size={22} />
          }
          accent="teal"
        />

        <KpiCard
          title="測試集 Macro F1"
          value={
            `${
              data
                .overview
                .test_macro_f1_percentage
                .toFixed(2)
            }%`
          }
          subtitle="保留 Holdout Test 最終評估"
          icon={
            <TrendingUp size={22} />
          }
          accent="teal"
        />

      </section>


      <section className="overview-grid">

        <DefectDistributionPanel
          data={
            data.distribution
          }
        />

        <DefectCompositionPanel
          data={
            data.distribution
          }
        />

        <ModelPerformancePanel
          data={
            data
              .performance
              .per_class
          }
        />

        <GlobalDriversPanel
          data={
            data.globalDrivers
          }
        />

      </section>


      <DefectIntelligenceSection />

    </div>
  );
}