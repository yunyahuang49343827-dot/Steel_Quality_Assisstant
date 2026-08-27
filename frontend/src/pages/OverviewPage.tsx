import {
  useEffect,
  useState,
} from "react";

import {
  AlertTriangle,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";

import {
  getOverviewDashboardData,
} from "../api/dashboard";

import OverviewDashboard
  from "../components/OverviewDashboard";

import type {
  OverviewDashboardData,
} from "../types/dashboard";

export default function OverviewPage() {

  const [
    data,
    setData,
  ] = useState<
    OverviewDashboardData | null
  >(null);

  const [
    loading,
    setLoading,
  ] = useState(true);

  const [
    error,
    setError,
  ] = useState<string | null>(null);

  async function loadDashboard() {

    try {

      setLoading(true);
      setError(null);

      const result =
        await getOverviewDashboardData();

      setData(result);

    } catch (err) {

      console.error(err);

      setError(
        "Unable to load dashboard data. Confirm the FastAPI backend is running on port 8000."
      );

    } finally {

      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  if (loading) {

    return (
      <div className="state-card">

        <LoaderCircle
          className="spin"
          size={30}
        />

        <h2>
          Loading Quality Intelligence
        </h2>

        <p>
          Retrieving PostgreSQL analytics,
          model evaluation and SHAP evidence.
        </p>

      </div>
    );
  }

  if (
    error ||
    !data
  ) {

    return (
      <div className="state-card error-state">

        <AlertTriangle
          size={30}
        />

        <h2>
          Dashboard unavailable
        </h2>

        <p>
          {error}
        </p>

        <button
          className="primary-button"
          onClick={
            loadDashboard
          }
        >

          <RefreshCw size={16} />
          Retry

        </button>

      </div>
    );
  }

  return (
    <OverviewDashboard
      data={data}
    />
  );
}