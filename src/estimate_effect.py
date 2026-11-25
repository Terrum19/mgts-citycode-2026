"""Illustrative on-premise economics, NOT observed MGTS performance or prices.

python src/estimate_effect.py
python src/estimate_effect.py --variant det_dino --seconds-per-photo 3.2

All defaults are replaceable assumptions. Server rates represent incremental electricity for existing hardware;
for a dedicated server set --fixed-server-month-rub. No OpenRouter/token fees.
"""
import argparse
import json


# seconds/photo and allocated RUB/server-hour; throughput is assumed, not measured.
VARIANTS = {
    "photo_dino": (0.2, 2.88),
    "boxes_dino": (0.4, 2.88),
    "det_dino": (0.5, 2.88),
    "vlm_8b": (8.0, 2.88),
    "det_vlm_8b": (12.0, 2.88),
    "cascade_8b": (5.25, 2.88),
    "vlm_235b": (8.0, 19.2),
    "det_vlm_235b": (12.0, 19.2),
    "cascade_235b": (5.25, 19.2),
}
SCENARIOS = {
    "conservative": (0.30, 0.05),
    "base": (0.15, 0.03),
    "optimistic": (0.08, 0.02),
}


def calculate(n, manual_seconds, review_seconds, wage, review_share, audit_share,
              seconds, server_rate, support, investment, realization=1.0,
              fixed_server=None):
    h0 = n * manual_seconds / 3600
    h1 = n * (review_share * review_seconds
              + (1 - review_share) * audit_share * manual_seconds) / 3600
    saved = h0 - h1
    machine_hours = n * seconds / 3600
    compute = machine_hours * server_rate if fixed_server is None else fixed_server
    benefit = saved * wage * realization
    opex = compute + support
    net = benefit - opex
    return {
        "manual_hours_before": h0, "manual_hours_after": h1,
        "hours_saved": saved, "labor_reduction": saved / h0,
        "photos_without_review_share": (1 - review_share) * (1 - audit_share),
        "assumed_seconds_per_photo": seconds, "photos_per_second": 1 / seconds,
        "speedup_vs_manual": manual_seconds / seconds,
        "machine_hours_per_month": machine_hours,
        "compute_rub_per_month": compute, "additional_opex_rub": opex,
        "labor_value_saved_rub": saved * wage,
        "realized_benefit_rub": benefit, "net_monthly_rub": net,
        "first_year_net_rub": 12 * net - investment,
        "roi_first_year": ((12 * net - investment) / (investment + 12 * opex)
                           if investment + 12 * opex > 0 else None),
        "simple_payback_months": investment / net if investment > 0 and net > 0 else None,
        "full_operating_rub_per_photo": (h1 * wage + opex) / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--photos-month", type=int, default=100000)
    p.add_argument("--manual-seconds", type=float, default=12)
    p.add_argument("--review-seconds", type=float, default=25)
    p.add_argument("--wage-hour-rub", type=float, default=650)
    p.add_argument("--support-month-rub", type=float, default=0)
    p.add_argument("--investment-rub", type=float, default=0)
    p.add_argument("--realization", type=float, default=1)
    p.add_argument("--review-share", type=float, help="override all scenarios with measured share")
    p.add_argument("--audit-share", type=float, help="override all scenarios with chosen audit share")
    p.add_argument("--seconds-per-photo", type=float)
    p.add_argument("--server-hour-rub", type=float)
    p.add_argument("--fixed-server-month-rub", type=float)
    a = p.parse_args()
    if a.photos_month <= 0 or a.manual_seconds <= 0 or a.review_seconds < 0:
        p.error("positive photo count/manual time and nonnegative review time required")
    if not 0 <= a.realization <= 1:
        p.error("realization must be in [0, 1]")
    for value in (a.review_share, a.audit_share):
        if value is not None and not 0 <= value <= 1:
            p.error("review/audit shares must be in [0, 1]")
    if a.seconds_per_photo is not None and a.seconds_per_photo <= 0:
        p.error("seconds-per-photo must be positive")
    for value in (a.wage_hour_rub, a.support_month_rub, a.investment_rub,
                  a.server_hour_rub, a.fixed_server_month_rub):
        if value is not None and value < 0:
            p.error("costs must be nonnegative")
    rows = {}
    for name, (seconds, rate) in VARIANTS.items():
        if a.variant and name != a.variant:
            continue
        rows[name] = {}
        for scenario, (q, audit) in SCENARIOS.items():
            q = q if a.review_share is None else a.review_share
            audit = audit if a.audit_share is None else a.audit_share
            rows[name][scenario] = calculate(
                a.photos_month, a.manual_seconds, a.review_seconds, a.wage_hour_rub,
                q, audit, a.seconds_per_photo or seconds,
                rate if a.server_hour_rub is None else a.server_hour_rub,
                a.support_month_rub, a.investment_rub, a.realization,
                a.fixed_server_month_rub)
    print(json.dumps({"status": "ILLUSTRATIVE_ASSUMPTIONS_NOT_MEASUREMENTS",
                      "inputs": vars(a), "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
