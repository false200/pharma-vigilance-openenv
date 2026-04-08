TASK_DATA = {
    "known_signal_easy": {
        "reports": [
            {
                "report_id": "PV-EASY-001",
                "patient_age": 59,
                "patient_sex": "female",
                "drugs": ["Lisinopril 20mg"],
                "suspect_drug": "Lisinopril",
                "reaction": "Persistent dry cough",
                "onset_days": 11,
                "severity": "mild",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 1264,
            }
        ],
        "ground_truth": {
            "classification": "known_side_effect",
            "suspect_drug": "Lisinopril",
            "severity_assessment": "mild",
            "recommended_action": "log_and_monitor",
        },
        "drug_interaction_db": {
            "Lisinopril": {
                "known_reactions": ["dry cough", "hyperkalemia", "angioedema"],
                "class_note": "ACE inhibitors frequently cause persistent non-productive cough.",
            }
        },
    },
    "cluster_signal_medium": {
        "reports": [
            {
                "report_id": "PV-MED-001",
                "patient_age": 44,
                "patient_sex": "female",
                "drugs": ["Cardiovexa"],
                "suspect_drug": "Cardiovexa",
                "reaction": "symptomatic bradycardia with dizziness",
                "onset_days": 9,
                "severity": "moderate",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 5,
            },
            {
                "report_id": "PV-MED-002",
                "patient_age": 69,
                "patient_sex": "male",
                "drugs": ["Cardiovexa"],
                "suspect_drug": "Cardiovexa",
                "reaction": "heart rate 32 with near-syncope",
                "onset_days": 13,
                "severity": "severe",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 5,
            },
            {
                "report_id": "PV-MED-003",
                "patient_age": 57,
                "patient_sex": "female",
                "drugs": ["Cardiovexa"],
                "suspect_drug": "Cardiovexa",
                "reaction": "fatigue and sinus bradycardia",
                "onset_days": 7,
                "severity": "moderate",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 5,
            },
            {
                "report_id": "PV-MED-004",
                "patient_age": 63,
                "patient_sex": "male",
                "drugs": ["Cardiovexa"],
                "suspect_drug": "Cardiovexa",
                "reaction": "bradyarrhythmia requiring ER evaluation",
                "onset_days": 11,
                "severity": "severe",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 5,
            },
        ],
        "ground_truth": {
            "classification": "new_signal",
            "suspect_drug": "Cardiovexa",
            "severity_assessment": "severe",
            "recommended_action": "escalate",
        },
        "drug_interaction_db": {
            "Cardiovexa": {
                "known_reactions": ["headache", "fatigue"],
                "approval_date": "5 months ago",
                "label_note": "No labeled conduction or rhythm adverse effects.",
            }
        },
    },
    "confounded_hard": {
        "reports": [
            {
                "report_id": "PV-HARD-001",
                "patient_age": 63,
                "patient_sex": "male",
                "drugs": [
                    "Tacrolimus",
                    "Prednisone",
                    "Amlodipine",
                    "Magnesium oxide",
                    "Voriconazole",
                    "Trimethoprim-sulfamethoxazole",
                ],
                "suspect_drug": "Trimethoprim-sulfamethoxazole",
                "reaction": "Acute kidney injury with tacrolimus trough 4x baseline",
                "onset_days": 6,
                "severity": "critical",
                "outcome": "not_recovered",
                "similar_reports_last_30d": 1,
            }
        ],
        "ground_truth": {
            "classification": "new_signal",
            "suspect_drug": "Tacrolimus+Voriconazole",
            "severity_assessment": "critical",
            "recommended_action": "escalate",
        },
        "drug_interaction_db": {
            "Voriconazole": {
                "strong_metabolic_inhibitor": True,
                "interacts_with": ["Tacrolimus", "Cyclosporine"],
                "interaction_note": "Markedly increases tacrolimus exposure; dose reduction and level monitoring required.",
            },
            "Tacrolimus": {
                "narrow_therapeutic_index": True,
                "known_reactions": ["nephrotoxicity", "tremor"],
                "requires_level_monitoring": True,
            },
        },
    },
}
