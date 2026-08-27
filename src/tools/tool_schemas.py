TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_quality_overview",
            "description": (
                "Get a high-level overview of the steel "
                "quality modeling dataset. Use this for "
                "questions about total sample count or "
                "number of defect classes."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_defect_distribution",
            "description": (
                "Get steel defect category counts and "
                "percentages. Use this for questions about "
                "the most common defect, least common defect, "
                "frequency, distribution, class balance, "
                "or number of records for a defect category."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_high_confidence_predictions",
            "description": (
                "Get a bounded list of model predictions "
                "with the highest prediction confidence. "
                "Use this only for questions about "
                "high-confidence predictions. "
                "Prediction confidence represents model "
                "certainty only and does not represent "
                "manufacturing risk, defect severity, "
                "or business impact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Number of predictions to return."
                        ),
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
                "additionalProperties": False,
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_defect_drivers",
            "description": (
                "Get the top SHAP predictive model drivers "
                "for one steel defect category. Use this "
                "when the user asks which features are "
                "important to the model for a specific "
                "defect class. These are predictive model "
                "drivers only and are not confirmed "
                "manufacturing root causes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "enum": [
                            "Bumps",
                            "Dirtiness",
                            "K_Scatch",
                            "Other_Faults",
                            "Pastry",
                            "Stains",
                            "Z_Scratch",
                        ],
                    },

                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": [
                    "defect_type"
                ],
                "additionalProperties": False,
            },
        },
    },
]