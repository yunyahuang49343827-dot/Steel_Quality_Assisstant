# Steel Quality Data Dictionary

## Dataset Structure

The dataset contains:

- 1 sample identifier
- 27 predictor features
- 7 binary defect target columns

The primary machine-learning task will later convert eligible records into a single-label multiclass classification dataset.

## Important Interpretation Note

Several variables are derived geometric or image-related indicators. Public documentation does not provide full manufacturing definitions for every derived index, so these variables are described conservatively and should not be interpreted as causal production parameters.

## Columns

| Column | Type | Role | Group | Description | Model Usage |
|---|---|---|---|---|---|
| id | int64 | Identifier | Identifier | Unique sample identifier provided by the Kaggle dataset. | Excluded from model training |
| X_Minimum | int64 | Feature | Geometry / Position | Minimum x-coordinate associated with the detected fault region. | Predictor |
| X_Maximum | int64 | Feature | Geometry / Position | Maximum x-coordinate associated with the detected fault region. | Predictor |
| Y_Minimum | int64 | Feature | Geometry / Position | Minimum y-coordinate associated with the detected fault region. | Predictor |
| Y_Maximum | int64 | Feature | Geometry / Position | Maximum y-coordinate associated with the detected fault region. | Predictor |
| Pixels_Areas | int64 | Feature | Geometry / Position | Area of the detected fault region measured in pixels. | Predictor |
| X_Perimeter | int64 | Feature | Geometry / Position | Horizontal perimeter-related measurement of the fault region. | Predictor |
| Y_Perimeter | int64 | Feature | Geometry / Position | Vertical perimeter-related measurement of the fault region. | Predictor |
| Sum_of_Luminosity | int64 | Feature | Luminosity | Sum of pixel luminosity values within the fault region. | Predictor |
| Minimum_of_Luminosity | int64 | Feature | Luminosity | Minimum observed luminosity value within the fault region. | Predictor |
| Maximum_of_Luminosity | int64 | Feature | Luminosity | Maximum observed luminosity value within the fault region. | Predictor |
| Length_of_Conveyer | int64 | Feature | Steel / Production | Conveyor length-related variable provided in the dataset. | Predictor |
| TypeOfSteel_A300 | int64 | Feature | Steel / Production | Binary indicator representing steel type A300. | Predictor |
| TypeOfSteel_A400 | int64 | Feature | Steel / Production | Binary indicator representing steel type A400. | Predictor |
| Steel_Plate_Thickness | int64 | Feature | Steel / Production | Steel plate thickness measurement provided by the dataset. | Predictor |
| Edges_Index | float64 | Feature | Shape / Edge Index | Derived edge-related index describing fault geometry. | Predictor |
| Empty_Index | float64 | Feature | Shape / Edge Index | Derived index related to empty space within the fault region. | Predictor |
| Square_Index | float64 | Feature | Shape / Edge Index | Derived index describing how square-like the fault region is. | Predictor |
| Outside_X_Index | float64 | Feature | Shape / Edge Index | Derived horizontal outside-region index. | Predictor |
| Edges_X_Index | float64 | Feature | Shape / Edge Index | Derived index describing horizontal edge characteristics. | Predictor |
| Edges_Y_Index | float64 | Feature | Shape / Edge Index | Derived index describing vertical edge characteristics. | Predictor |
| Outside_Global_Index | float64 | Feature | Shape / Edge Index | Derived global outside-region index. | Predictor |
| LogOfAreas | float64 | Feature | Log Transformation | Log-transformed representation of fault area. | Predictor |
| Log_X_Index | float64 | Feature | Log Transformation | Log-transformed horizontal fault-related index. | Predictor |
| Log_Y_Index | float64 | Feature | Log Transformation | Log-transformed vertical fault-related index. | Predictor |
| Orientation_Index | float64 | Feature | Shape / Edge Index | Derived index describing orientation characteristics of the fault region. | Predictor |
| Luminosity_Index | float64 | Feature | Luminosity | Derived luminosity-related index describing the fault region. | Predictor |
| SigmoidOfAreas | float64 | Feature | Area Transformation | Sigmoid-transformed representation of fault area. | Predictor |
| Pastry | int64 | Target | Defect Target | Binary indicator for the Pastry defect category. | Used to construct multiclass target |
| Z_Scratch | int64 | Target | Defect Target | Binary indicator for the Z_Scratch defect category. | Used to construct multiclass target |
| K_Scatch | int64 | Target | Defect Target | Binary indicator for the K_Scatch defect category. | Used to construct multiclass target |
| Stains | int64 | Target | Defect Target | Binary indicator for the Stains defect category. | Used to construct multiclass target |
| Dirtiness | int64 | Target | Defect Target | Binary indicator for the Dirtiness defect category. | Used to construct multiclass target |
| Bumps | int64 | Target | Defect Target | Binary indicator for the Bumps defect category. | Used to construct multiclass target |
| Other_Faults | int64 | Target | Defect Target | Binary indicator for defects grouped under Other_Faults. | Used to construct multiclass target |

## Target Categories

- `Pastry`
- `Z_Scratch`
- `K_Scatch`
- `Stains`
- `Dirtiness`
- `Bumps`
- `Other_Faults`

## Modeling Note

`id` is used only as a record identifier and will not be included as a predictive feature.

Samples without exactly one positive target label will be handled as data-quality exceptions during Stage B3.
