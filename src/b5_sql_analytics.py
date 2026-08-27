from pathlib import Path
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "sql_analytics"
)

DOCS_DIR = PROJECT_ROOT / "docs"

ENV_PATH = PROJECT_ROOT / ".env"

MARKDOWN_OUTPUT_PATH = (
    DOCS_DIR / "sql_analytics_report.md"
)


# =========================================================
# 2. Environment configuration
# =========================================================

load_dotenv(ENV_PATH)

DB_CONFIG = {
    "host": os.getenv(
        "DB_HOST",
        "localhost",
    ),

    "port": os.getenv(
        "DB_PORT",
        "5432",
    ),

    "dbname": os.getenv(
        "DB_NAME",
        "steel_quality",
    ),

    "user": os.getenv(
        "DB_USER"
    ),

    "password": os.getenv(
        "DB_PASSWORD",
        "",
    ),
}


# =========================================================
# 3. SQL Queries
# =========================================================

DEFECT_DISTRIBUTION_QUERY = """
SELECT
    defect_type,
    COUNT(*) AS sample_count,
    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage
FROM modeling_steel_quality
GROUP BY defect_type
ORDER BY sample_count DESC;
"""


DEFECT_FEATURE_SUMMARY_QUERY = """
SELECT
    defect_type,

    COUNT(*) AS sample_count,

    ROUND(
        AVG("Steel_Plate_Thickness")::numeric,
        2
    ) AS avg_plate_thickness,

    ROUND(
        AVG("Pixels_Areas")::numeric,
        2
    ) AS avg_pixels_area,

    ROUND(
        AVG("X_Perimeter")::numeric,
        2
    ) AS avg_x_perimeter,

    ROUND(
        AVG("Y_Perimeter")::numeric,
        2
    ) AS avg_y_perimeter,

    ROUND(
        AVG("Luminosity_Index")::numeric,
        4
    ) AS avg_luminosity_index

FROM modeling_steel_quality

GROUP BY defect_type

ORDER BY sample_count DESC;
"""


THICKNESS_ANALYSIS_QUERY = """
WITH thickness_groups AS (

    SELECT
        defect_type,

        "Steel_Plate_Thickness",

        CASE

            WHEN "Steel_Plate_Thickness" < 70
                THEN 'Thin'

            WHEN "Steel_Plate_Thickness" < 150
                THEN 'Medium'

            ELSE 'Thick'

        END AS thickness_group

    FROM modeling_steel_quality
)

SELECT
    thickness_group,
    defect_type,
    COUNT(*) AS sample_count

FROM thickness_groups

GROUP BY
    thickness_group,
    defect_type

ORDER BY
    thickness_group,
    sample_count DESC;
"""


LUMINOSITY_ANALYSIS_QUERY = """
SELECT
    defect_type,

    COUNT(*) AS sample_count,

    ROUND(
        AVG("Minimum_of_Luminosity")::numeric,
        2
    ) AS avg_min_luminosity,

    ROUND(
        AVG("Maximum_of_Luminosity")::numeric,
        2
    ) AS avg_max_luminosity,

    ROUND(
        AVG("Luminosity_Index")::numeric,
        4
    ) AS avg_luminosity_index

FROM modeling_steel_quality

GROUP BY defect_type

ORDER BY avg_luminosity_index;
"""


AREA_ANALYSIS_QUERY = """
SELECT
    defect_type,

    COUNT(*) AS sample_count,

    ROUND(
        AVG("Pixels_Areas")::numeric,
        2
    ) AS avg_pixels_area,

    ROUND(
        MIN("Pixels_Areas")::numeric,
        2
    ) AS min_pixels_area,

    ROUND(
        MAX("Pixels_Areas")::numeric,
        2
    ) AS max_pixels_area,

    ROUND(
        AVG("LogOfAreas")::numeric,
        4
    ) AS avg_log_area

FROM modeling_steel_quality

GROUP BY defect_type

ORDER BY avg_pixels_area DESC;
"""


# =========================================================
# 4. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Validate database configuration.
    """

    if not DB_CONFIG["user"]:

        raise ValueError(
            "DB_USER is missing. "
            "Please configure the .env file."
        )


# =========================================================
# 5. Database connection
# =========================================================

def get_connection():
    """
    Create PostgreSQL connection.
    """

    connection_kwargs = {
        "host": DB_CONFIG["host"],
        "port": DB_CONFIG["port"],
        "dbname": DB_CONFIG["dbname"],
        "user": DB_CONFIG["user"],
    }

    if DB_CONFIG["password"]:

        connection_kwargs[
            "password"
        ] = DB_CONFIG["password"]

    return psycopg.connect(
        **connection_kwargs
    )


# =========================================================
# 6. Execute SQL query
# =========================================================

def execute_query(
    conn,
    query: str,
) -> pd.DataFrame:
    """
    Execute a SQL query and return the result
    as a pandas DataFrame.
    """

    with conn.cursor() as cursor:

        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [
            description.name
            for description
            in cursor.description
        ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# =========================================================
# 7. Run analytics
# =========================================================

def run_sql_analytics(
    conn,
) -> dict:
    """
    Execute all Stage B5 analytics queries.
    """

    results = {}

    results[
        "defect_distribution"
    ] = execute_query(
        conn,
        DEFECT_DISTRIBUTION_QUERY,
    )

    results[
        "defect_feature_summary"
    ] = execute_query(
        conn,
        DEFECT_FEATURE_SUMMARY_QUERY,
    )

    results[
        "thickness_analysis"
    ] = execute_query(
        conn,
        THICKNESS_ANALYSIS_QUERY,
    )

    results[
        "luminosity_analysis"
    ] = execute_query(
        conn,
        LUMINOSITY_ANALYSIS_QUERY,
    )

    results[
        "area_analysis"
    ] = execute_query(
        conn,
        AREA_ANALYSIS_QUERY,
    )

    return results


# =========================================================
# 8. Save CSV outputs
# =========================================================

def save_csv_outputs(
    results: dict,
) -> None:
    """
    Save each SQL result as a CSV report.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, dataframe in (
        results.items()
    ):

        output_path = (
            REPORTS_DIR
            / f"{name}.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )


# =========================================================
# 9. Helper: Markdown table
# =========================================================

def dataframe_to_markdown(
    df: pd.DataFrame,
) -> str:
    """
    Convert a dataframe to a simple Markdown table
    without external dependencies.
    """

    columns = list(df.columns)

    lines = []

    header = (
        "| "
        + " | ".join(columns)
        + " |"
    )

    separator = (
        "| "
        + " | ".join(
            ["---"] * len(columns)
        )
        + " |"
    )

    lines.append(header)
    lines.append(separator)

    for _, row in df.iterrows():

        values = [
            str(value)
            for value in row
        ]

        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )

    return "\n".join(lines)


# =========================================================
# 10. Generate Markdown report
# =========================================================

def generate_markdown_report(
    results: dict,
) -> str:
    """
    Generate human-readable SQL Analytics documentation.
    """

    defect_distribution = (
        results[
            "defect_distribution"
        ]
    )

    feature_summary = (
        results[
            "defect_feature_summary"
        ]
    )

    thickness_analysis = (
        results[
            "thickness_analysis"
        ]
    )

    luminosity_analysis = (
        results[
            "luminosity_analysis"
        ]
    )

    area_analysis = (
        results[
            "area_analysis"
        ]
    )

    most_common_defect = (
        defect_distribution.iloc[0]
    )

    largest_area_defect = (
        area_analysis.iloc[0]
    )

    lines = [
        "# SQL Quality Analytics Report",
        "",
        "## Purpose",
        "",
        (
            "This stage uses PostgreSQL to perform "
            "descriptive quality analytics on the "
            "model-ready steel defect dataset."
        ),
        "",
        (
            "SQL is used as the factual analytics layer. "
            "The results will later support FastAPI "
            "endpoints and LLM function-calling tools."
        ),
        "",
        "## 1. Defect Distribution",
        "",
        dataframe_to_markdown(
            defect_distribution
        ),
        "",
        "### Key Observation",
        "",
        (
            f"The most common defect category is "
            f"`{most_common_defect['defect_type']}` "
            f"with {most_common_defect['sample_count']} "
            f"samples "
            f"({most_common_defect['percentage']}%)."
        ),
        "",
        "## 2. Defect Feature Summary",
        "",
        dataframe_to_markdown(
            feature_summary
        ),
        "",
        (
            "These values describe average feature "
            "characteristics by defect category. "
            "They should not be interpreted as "
            "causal manufacturing relationships."
        ),
        "",
        "## 3. Thickness Group Analysis",
        "",
        dataframe_to_markdown(
            thickness_analysis
        ),
        "",
        "### Important Note",
        "",
        (
            "Thin / Medium / Thick are exploratory "
            "data segmentation groups created for "
            "this project. They are not presented "
            "as official manufacturing thickness "
            "standards."
        ),
        "",
        "## 4. Luminosity Analysis",
        "",
        dataframe_to_markdown(
            luminosity_analysis
        ),
        "",
        "## 5. Fault Area Analysis",
        "",
        dataframe_to_markdown(
            area_analysis
        ),
        "",
        "### Key Observation",
        "",
        (
            f"`{largest_area_defect['defect_type']}` "
            f"has the highest average Pixels_Areas "
            f"in this dataset, with an average of "
            f"{largest_area_defect['avg_pixels_area']}."
        ),
        "",
        "## Interpretation Principle",
        "",
        (
            "SQL analytics describe patterns observed "
            "in the available dataset. Differences "
            "between defect classes do not establish "
            "causal process relationships."
        ),
        "",
    ]

    return "\n".join(lines)


# =========================================================
# 11. Save Markdown
# =========================================================

def save_markdown_report(
    markdown_text: str,
) -> None:
    """
    Save SQL analytics documentation.
    """

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown_text,
        encoding="utf-8",
    )


# =========================================================
# 12. Validate results
# =========================================================

def validate_results(
    results: dict,
) -> None:
    """
    Perform basic sanity checks on SQL outputs.
    """

    defect_distribution = (
        results[
            "defect_distribution"
        ]
    )

    total_samples = int(
        defect_distribution[
            "sample_count"
        ].sum()
    )

    if total_samples != 18380:

        raise ValueError(
            "Unexpected total number "
            "of modeling samples: "
            f"{total_samples}"
        )

    if len(defect_distribution) != 7:

        raise ValueError(
            "Expected 7 defect classes, "
            f"found {len(defect_distribution)}."
        )

    required_results = [
        "defect_distribution",
        "defect_feature_summary",
        "thickness_analysis",
        "luminosity_analysis",
        "area_analysis",
    ]

    for result_name in (
        required_results
    ):

        if results[
            result_name
        ].empty:

            raise ValueError(
                f"{result_name} returned "
                "no rows."
            )


# =========================================================
# 13. Print results
# =========================================================

def print_summary(
    results: dict,
) -> None:

    print("=" * 72)
    print(
        "Stage B5 — SQL Quality Analytics"
    )
    print("=" * 72)

    print(
        "\nDEFECT DISTRIBUTION"
    )
    print("-" * 72)

    print(
        results[
            "defect_distribution"
        ].to_string(
            index=False
        )
    )

    print(
        "\nDEFECT FEATURE SUMMARY"
    )
    print("-" * 72)

    print(
        results[
            "defect_feature_summary"
        ].to_string(
            index=False
        )
    )

    print(
        "\nLUMINOSITY ANALYSIS"
    )
    print("-" * 72)

    print(
        results[
            "luminosity_analysis"
        ].to_string(
            index=False
        )
    )

    print(
        "\nFAULT AREA ANALYSIS"
    )
    print("-" * 72)

    print(
        results[
            "area_analysis"
        ].to_string(
            index=False
        )
    )

    print(
        "\nOUTPUT FILES"
    )
    print("-" * 72)

    print(
        f"Reports  : {REPORTS_DIR}"
    )

    print(
        f"Markdown : "
        f"{MARKDOWN_OUTPUT_PATH}"
    )

    print(
        "\nSQL Analytics validation: PASSED"
    )

    print("=" * 72)


# =========================================================
# 14. Main
# =========================================================

def main() -> None:

    validate_configuration()

    with get_connection() as conn:

        results = (
            run_sql_analytics(
                conn
            )
        )

    validate_results(
        results
    )

    save_csv_outputs(
        results
    )

    markdown_report = (
        generate_markdown_report(
            results
        )
    )

    save_markdown_report(
        markdown_report
    )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()