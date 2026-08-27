-- =========================================================
-- Steel Quality Intelligence
-- PostgreSQL Docker Seed
-- =========================================================


\echo '=========================================================='
\echo 'Loading Steel Quality datasets'
\echo '=========================================================='


-- ---------------------------------------------------------
-- 1. Raw Kaggle training dataset
-- ---------------------------------------------------------

COPY raw_steel_quality
FROM '/seed-data/train.csv'
WITH (
    FORMAT csv,
    HEADER true
);


-- ---------------------------------------------------------
-- 2. Modeling dataset
-- ---------------------------------------------------------

COPY modeling_steel_quality
FROM '/seed-data/steel_quality_modeling.csv'
WITH (
    FORMAT csv,
    HEADER true
);


-- ---------------------------------------------------------
-- 3. PostgreSQL planner statistics
-- ---------------------------------------------------------

ANALYZE raw_steel_quality;
ANALYZE modeling_steel_quality;


-- ---------------------------------------------------------
-- 4. Initialization validation
-- ---------------------------------------------------------

DO $$
DECLARE
    raw_count BIGINT;
    modeling_count BIGINT;
    defect_count BIGINT;
BEGIN

    SELECT COUNT(*)
    INTO raw_count
    FROM raw_steel_quality;

    SELECT COUNT(*)
    INTO modeling_count
    FROM modeling_steel_quality;

    SELECT COUNT(
        DISTINCT defect_type
    )
    INTO defect_count
    FROM modeling_steel_quality;


    IF raw_count <> 19219 THEN

        RAISE EXCEPTION
            'raw_steel_quality validation failed: expected 19219 rows, got %',
            raw_count;

    END IF;


    IF modeling_count <> 18380 THEN

        RAISE EXCEPTION
            'modeling_steel_quality validation failed: expected 18380 rows, got %',
            modeling_count;

    END IF;


    IF defect_count <> 7 THEN

        RAISE EXCEPTION
            'defect class validation failed: expected 7 classes, got %',
            defect_count;

    END IF;


    RAISE NOTICE
        'Database seed validation passed: raw=%, modeling=%, defect_classes=%',
        raw_count,
        modeling_count,
        defect_count;

END
$$;


\echo '=========================================================='
\echo 'Steel Quality database initialization complete'
\echo '=========================================================='