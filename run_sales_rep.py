"""
Entry point wrapper: imports run_sales_rep_flow from agent_loop (which handles logging).
"""

from typing import Dict, Optional

from agent_loop import run_sales_rep_flow


if __name__ == "__main__":
    MY_COMPANY = (
        "K2X Technologies: We provide AI-driven software solutions for industrial companies "
        "to improve operational efficiency and reduce downtime."
    )
    PROSPECT_COMPANY = "Antonx Private Limited"
    PROSPECT_INDUSTRY = "Software solutions"
    PROSPECT_PROFILE = (
        "antonx.com"
    )

    out = run_sales_rep_flow(
        my_company_description=MY_COMPANY,
        prospect_company_name=PROSPECT_COMPANY,
        prospect_industry=PROSPECT_INDUSTRY,
        prospect_profile_text=PROSPECT_PROFILE,
        max_steps=5,
        run_initial_search=True,
    )
    print("VALUE HYPOTHESIS:", out.get("value_hypothesis", ""))
    print("MESSAGING ANGLE:", out.get("messaging_angle", ""))
    print("SUPPORTING EVIDENCE:", out.get("supporting_evidence", ""))
