"""Fila curta e sequencial de hipóteses predefinidas; não é busca automática."""

import argparse
import json
from pathlib import Path

from campaign import collect, run

from airline_ml.utils import utc_now, write_json

STAGES = {
    "B": ["camp_cb_depth5_s500", "camp_cb_depth7_s500", "camp_cb_l2_10_s500", "camp_cb_lr08_s320"],
    "CD": [
        "camp_lgb_service_categorical_s800",
        "camp_lgb_delays_s800",
        "camp_lgb_rating_counts_s800",
        "camp_lgb_rating_summary_s800",
        "camp_lgb_rating_differences_s800",
        "camp_lgb_segments_s800",
        "camp_lgb_service_segments_s800",
        "camp_lgb_leaves63_s800",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    names = STAGES[args.stage]
    for index, name in enumerate(names):
        pointer = Path("artifacts/campaign") / (name + "_screen_result.json")
        if pointer.exists():
            continue
        write_json(
            "reports/campaign_stage_status.json",
            {
                "stage": args.stage,
                "current": name,
                "index": index + 1,
                "total": len(names),
                "started_at_utc": utc_now(),
                "status": "running",
            },
        )
        run(f"configs/campaign/{name}.json", [0], 1200)
        collect()
    write_json(
        "reports/campaign_stage_status.json",
        {
            "stage": args.stage,
            "completed": names,
            "finished_at_utc": utc_now(),
            "status": "completed",
        },
    )
    print(json.dumps({"stage": args.stage, "trials": len(names), "status": "completed"}))
