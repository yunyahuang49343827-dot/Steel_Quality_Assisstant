from pathlib import Path
import os

import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = DATA_DIR / "train.csv"

MODELING_DATA_PATH = (
    PROCESSED_DIR / "steel_quality_modeling.csv"
)

ENV_PATH = PROJECT_ROOT / ".env"


# =========================================================
# 2. Load environment variables
# =========================================================

load_dotenv(ENV_PATH)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv(
        "DB_NAME",
        "steel_quality",
    ),
    "user": os.getenv("DB_USER"),
    "password": os.getenv(
        "DB_PASSWORD",
        "",
    ),
}


# =========================================================
# 3. Table names
# =========================================================

RAW_TABLE = "raw_steel_quality"

MODELING_TABLE = "modeling_steel_quality"


# =========================================================
# 4. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Confirm that required database configuration and
    source files are available.
    """

    if not DB_CONFIG["user"]:
        raise ValueError(
            "DB_USER is missing. "
            "Please configure it in the .env file."
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed modeling dataset not found:\n"
            f"{MODELING_DATA_PATH}\n"
            "Please complete Stage B3 first."
        )


# =========================================================
# 5. Connect to PostgreSQL
# =========================================================

def get_connection():
    """
    Create a PostgreSQL database connection.
    """

    connection_kwargs = {
        "host": DB_CONFIG["host"],
        "port": DB_CONFIG["port"],
        "dbname": DB_CONFIG["dbname"],
        "user": DB_CONFIG["user"],
    }

    if DB_CONFIG["password"]:
        connection_kwargs["password"] = (
            DB_CONFIG["password"]
        )

    return psycopg.connect(
        **connection_kwargs
    )


# =========================================================
# 6. Load source CSV files
# =========================================================

def load_source_data():
    """
    Load raw and processed datasets from CSV.
    """

    raw_df = pd.read_csv(
        RAW_DATA_PATH
    )

    modeling_df = pd.read_csv(
        MODELING_DATA_PATH
    )

    return raw_df, modeling_df


# =========================================================
# 7. PostgreSQL type mapping
# =========================================================

def pandas_dtype_to_postgres(
    dtype,
) -> str:
    """
    Convert common pandas dtypes to PostgreSQL types.
    """

    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"

    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"

    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"

    return "TEXT"


# =========================================================
# 8. Create table from dataframe schema
# =========================================================

def create_table(
    conn,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Recreate a PostgreSQL table using dataframe columns
    and dtypes.

    The table is dropped and rebuilt to make this Stage
    reproducible during development.
    """

    column_definitions = []

    for column in df.columns:

        postgres_type = (
            pandas_dtype_to_postgres(
                df[column].dtype
            )
        )

        column_definitions.append(
            sql.SQL("{} {}").format(
                sql.Identifier(column),
                sql.SQL(postgres_type),
            )
        )

    create_statement = sql.SQL(
        """
        CREATE TABLE {} (
            {}
        )
        """
    ).format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(
            column_definitions
        ),
    )

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                "DROP TABLE IF EXISTS {}"
            ).format(
                sql.Identifier(table_name)
            )
        )

        cur.execute(create_statement)


# =========================================================
# 9. Insert dataframe
# =========================================================

def insert_dataframe(
    conn,
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """
    Insert dataframe rows into PostgreSQL.

    execute_batch-style row insertion is sufficient for
    this small portfolio dataset.
    """

    columns = list(df.columns)

    placeholders = sql.SQL(", ").join(
        [sql.Placeholder()] * len(columns)
    )

    insert_statement = sql.SQL(
        """
        INSERT INTO {} ({})
        VALUES ({})
        """
    ).format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(
            map(
                sql.Identifier,
                columns,
            )
        ),
        placeholders,
    )

    clean_df = df.where(
        pd.notnull(df),
        None,
    )

    rows = [
        tuple(row)
        for row in clean_df.itertuples(
            index=False,
            name=None,
        )
    ]

    with conn.cursor() as cur:

        cur.executemany(
            insert_statement,
            rows,
        )


# =========================================================
# 10. Create indexes
# =========================================================

def create_indexes(
    conn,
) -> None:
    """
    Add indexes useful for future SQL/API queries.
    """

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                idx_raw_steel_quality_id
                ON {} ("id")
                """
            ).format(
                sql.Identifier(RAW_TABLE)
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                idx_modeling_steel_quality_id
                ON {} ("id")
                """
            ).format(
                sql.Identifier(
                    MODELING_TABLE
                )
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS
                idx_modeling_defect_type
                ON {} ("defect_type")
                """
            ).format(
                sql.Identifier(
                    MODELING_TABLE
                )
            )
        )


# =========================================================
# 11. Validate database counts
# =========================================================

def get_table_count(
    conn,
    table_name: str,
) -> int:
    """
    Return table row count.
    """

    with conn.cursor() as cur:

        cur.execute(
            sql.SQL(
                "SELECT COUNT(*) FROM {}"
            ).format(
                sql.Identifier(
                    table_name
                )
            )
        )

        return cur.fetchone()[0]


# =========================================================
# 12. Validate table schemas
# =========================================================

def get_table_column_count(
    conn,
    table_name: str,
) -> int:
    """
    Return number of columns in a PostgreSQL table.
    """

    query = """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = %s
    """

    with conn.cursor() as cur:

        cur.execute(
            query,
            (table_name,),
        )

        return cur.fetchone()[0]


# =========================================================
# 13. Database setup
# =========================================================

def setup_database(
    raw_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
):
    """
    Create and populate raw + modeling PostgreSQL tables.
    """

    with get_connection() as conn:

        print(
            "\nCreating raw table..."
        )

        create_table(
            conn,
            RAW_TABLE,
            raw_df,
        )

        insert_dataframe(
            conn,
            RAW_TABLE,
            raw_df,
        )

        print(
            "Creating modeling table..."
        )

        create_table(
            conn,
            MODELING_TABLE,
            modeling_df,
        )

        insert_dataframe(
            conn,
            MODELING_TABLE,
            modeling_df,
        )

        create_indexes(conn)

        raw_count = get_table_count(
            conn,
            RAW_TABLE,
        )

        modeling_count = get_table_count(
            conn,
            MODELING_TABLE,
        )

        raw_column_count = (
            get_table_column_count(
                conn,
                RAW_TABLE,
            )
        )

        modeling_column_count = (
            get_table_column_count(
                conn,
                MODELING_TABLE,
            )
        )

        return {
            "raw_rows": raw_count,
            "modeling_rows": (
                modeling_count
            ),
            "raw_columns": (
                raw_column_count
            ),
            "modeling_columns": (
                modeling_column_count
            ),
        }


# =========================================================
# 14. Validate against source
# =========================================================

def validate_database(
    raw_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
    db_summary: dict,
) -> None:
    """
    Ensure PostgreSQL contains exactly the expected rows
    and columns.
    """

    expected = {
        "raw_rows": len(raw_df),

        "modeling_rows":
            len(modeling_df),

        "raw_columns":
            len(raw_df.columns),

        "modeling_columns":
            len(modeling_df.columns),
    }

    for key, expected_value in (
        expected.items()
    ):

        actual_value = db_summary[key]

        if actual_value != expected_value:

            raise ValueError(
                f"Database validation failed "
                f"for {key}: "
                f"expected {expected_value}, "
                f"got {actual_value}"
            )


# =========================================================
# 15. Print summary
# =========================================================

def print_summary(
    db_summary: dict,
) -> None:

    print("=" * 72)
    print(
        "Stage B4 — PostgreSQL Data Layer"
    )
    print("=" * 72)

    print("\nDATABASE")
    print("-" * 72)

    print(
        f"Host       : "
        f"{DB_CONFIG['host']}"
    )

    print(
        f"Port       : "
        f"{DB_CONFIG['port']}"
    )

    print(
        f"Database   : "
        f"{DB_CONFIG['dbname']}"
    )

    print(
        f"User       : "
        f"{DB_CONFIG['user']}"
    )

    print("\nRAW TABLE")
    print("-" * 72)

    print(
        f"Table      : {RAW_TABLE}"
    )

    print(
        f"Rows       : "
        f"{db_summary['raw_rows']:,}"
    )

    print(
        f"Columns    : "
        f"{db_summary['raw_columns']}"
    )

    print("\nMODELING TABLE")
    print("-" * 72)

    print(
        f"Table      : "
        f"{MODELING_TABLE}"
    )

    print(
        f"Rows       : "
        f"{db_summary['modeling_rows']:,}"
    )

    print(
        f"Columns    : "
        f"{db_summary['modeling_columns']}"
    )

    print("\nINDEXES")
    print("-" * 72)

    print(
        "raw.id              : UNIQUE"
    )

    print(
        "modeling.id         : UNIQUE"
    )

    print(
        "modeling.defect_type: INDEXED"
    )

    print(
        "\nDatabase validation : PASSED"
    )

    print("=" * 72)


# =========================================================
# 16. Main
# =========================================================

def main():

    validate_configuration()

    raw_df, modeling_df = (
        load_source_data()
    )

    print("=" * 72)
    print(
        "Stage B4 — PostgreSQL Data Layer"
    )
    print("=" * 72)

    print(
        "\nConnecting to PostgreSQL..."
    )

    db_summary = setup_database(
        raw_df,
        modeling_df,
    )

    validate_database(
        raw_df,
        modeling_df,
        db_summary,
    )

    print_summary(
        db_summary
    )


if __name__ == "__main__":
    main()