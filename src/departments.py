# Maps each of the 24 resume categories to a broader department.
# Used AFTER the model predicts, so no retraining is needed.

DEPT_MAP = {
    "INFORMATION-TECHNOLOGY": "Technology & Engineering",
    "ENGINEERING": "Technology & Engineering",
    "AUTOMOBILE": "Technology & Engineering",
    "AVIATION": "Technology & Engineering",
    "CONSTRUCTION": "Technology & Engineering",
    "BUSINESS-DEVELOPMENT": "Business & Management",
    "CONSULTANT": "Business & Management",
    "SALES": "Business & Management",
    "HR": "Business & Management",
    "BPO": "Business & Management",
    "PUBLIC-RELATIONS": "Business & Management",
    "FINANCE": "Finance & Legal",
    "ACCOUNTANT": "Finance & Legal",
    "BANKING": "Finance & Legal",
    "ADVOCATE": "Finance & Legal",
    "HEALTHCARE": "Healthcare & Wellness",
    "FITNESS": "Healthcare & Wellness",
    "DESIGNER": "Creative & Media",
    "DIGITAL-MEDIA": "Creative & Media",
    "ARTS": "Creative & Media",
    "APPAREL": "Creative & Media",
    "TEACHER": "Education",
    "CHEF": "Hospitality & Agriculture",
    "AGRICULTURE": "Hospitality & Agriculture",
}

def get_department(category: str) -> str:
    """Return the department for a predicted category (case-insensitive)."""
    return DEPT_MAP.get(str(category).upper().strip(), "Other")