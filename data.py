TASK_DATA = {
    "known_signal_easy": {
        "reports": [
            {
                "report_id": "PV-EASY-001",
                "patient_age": 67,
                "patient_sex": "male",
                "drugs": ["Ibuprofen 800mg"],
                "suspect_drug": "Ibuprofen",
                "reaction": "Gastrointestinal bleeding",
                "onset_days": 3,
                "severity": "moderate",
                "outcome": "recovering",
                "similar_reports_last_30d": 847,
            }
        ],
        "ground_truth": {
            "classification": "known_side_effect",
            "suspect_drug": "Ibuprofen",
            "severity_assessment": "moderate",
            "recommended_action": "log_and_monitor",
        },
        "drug_interaction_db": {
            "Ibuprofen": {
                "known_reactions": ["GI bleeding", "stomach ulcer", "nausea"],
                "black_box_warning": "GI bleeding risk increases in older adults and higher-dose use.",
            }
        },
    },
    "cluster_signal_medium": {
        "reports": [
            {
                "report_id": "PV-MED-001",
                "patient_age": 52,
                "patient_sex": "female",
                "drugs": ["Gliptozin"],
                "suspect_drug": "Gliptozin",
                "reaction": "sudden blurred vision",
                "onset_days": 14,
                "severity": "mild",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 4,
            },
            {
                "report_id": "PV-MED-002",
                "patient_age": 61,
                "patient_sex": "male",
                "drugs": ["Gliptozin"],
                "suspect_drug": "Gliptozin",
                "reaction": "sudden vision loss left eye",
                "onset_days": 21,
                "severity": "severe",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 4,
            },
            {
                "report_id": "PV-MED-003",
                "patient_age": 48,
                "patient_sex": "female",
                "drugs": ["Gliptozin"],
                "suspect_drug": "Gliptozin",
                "reaction": "visual disturbance",
                "onset_days": 10,
                "severity": "mild",
                "outcome": "recovering",
                "similar_reports_last_30d": 4,
            },
            {
                "report_id": "PV-MED-004",
                "patient_age": 55,
                "patient_sex": "male",
                "drugs": ["Gliptozin"],
                "suspect_drug": "Gliptozin",
                "reaction": "partial vision loss",
                "onset_days": 18,
                "severity": "moderate",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 4,
            },
        ],
        "ground_truth": {
            "classification": "new_signal",
            "suspect_drug": "Gliptozin",
            "severity_assessment": "severe",
            "recommended_action": "escalate",
        },
        "drug_interaction_db": {
            "Gliptozin": {
                "known_reactions": ["hypoglycemia", "nausea"],
                "approval_date": "8 months ago",
            }
        },
    },
    "confounded_hard": {
        "reports": [
            {
                "report_id": "PV-HARD-001",
                "patient_age": 71,
                "patient_sex": "male",
                "drugs": [
                    "Warfarin",
                    "Metformin",
                    "Lisinopril",
                    "Atorvastatin",
                    "Clarithromycin",
                    "Omeprazole",
                ],
                "suspect_drug": "Omeprazole",
                "reaction": "Severe liver failure (ALT 10x normal)",
                "onset_days": 12,
                "severity": "critical",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 1,
            }
        ],
        "ground_truth": {
            "classification": "new_signal",
            "suspect_drug": "Atorvastatin+Clarithromycin",
            "severity_assessment": "critical",
            "recommended_action": "escalate",
        },
        "drug_interaction_db": {
            "Clarithromycin": {
                "cyp3a4_inhibitor": True,
                "interacts_with": ["Atorvastatin", "Warfarin"],
            },
            "Atorvastatin": {
                "cyp3a4_substrate": True,
                "known_reactions": ["myopathy"],
                "hepatotoxic_when_levels_elevated": True,
            },
        },
    },
}
