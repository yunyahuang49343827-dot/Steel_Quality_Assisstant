-- =========================================================
-- Steel Quality Prediction, Explainability & AI Analytics Copilot
-- Stage B5 — SQL Analytics
-- =========================================================


-- =========================================================
-- Query 1
-- Defect Distribution
-- =========================================================

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


-- =========================================================
-- Query 2
-- Defect Feature Summary
-- =========================================================

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


-- =========================================================
-- Query 3
-- Thickness Group Analysis
-- =========================================================

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


-- =========================================================
-- Query 4
-- Luminosity Analysis
-- =========================================================

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


-- =========================================================
-- Query 5
-- Fault Area Analysis
-- =========================================================

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