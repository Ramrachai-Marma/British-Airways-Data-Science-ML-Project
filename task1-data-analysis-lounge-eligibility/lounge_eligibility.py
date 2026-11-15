"""Build the lounge eligibility lookup table for Heathrow Terminal 3 (Task 1).

Eligibility % = eligible passengers / total seats, summed across each group.
Run from anywhere:  python task1-data-analysis-lounge-eligibility/lounge_eligibility.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ba_summer_schedule.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"

TIER_COLUMNS = {
    "tier1": "TIER1_ELIGIBLE_PAX",
    "tier2": "TIER2_ELIGIBLE_PAX",
    "tier3": "TIER3_ELIGIBLE_PAX",
}
SEAT_COLUMNS = ["FIRST_CLASS_SEATS", "BUSINESS_CLASS_SEATS", "ECONOMY_SEATS"]
SAMPLE_FLIGHTS = ["BA1896", "BA5211"]


def load_schedule(path=DATA_PATH):
    schedule = pd.read_excel(path)
    schedule["SEATS"] = schedule[SEAT_COLUMNS].sum(axis=1)
    schedule["HAUL"] = schedule["HAUL"].str.title() + "-haul"
    schedule["CATEGORY"] = schedule["HAUL"] + " - " + schedule["ARRIVAL_REGION"]
    return schedule


def eligibility_table(schedule, group_columns):
    """Eligible passengers as a share of seat capacity for each group."""
    grouped = schedule.groupby(group_columns)
    totals = grouped[["SEATS"] + list(TIER_COLUMNS.values())].sum()
    table = pd.DataFrame(
        {
            "flights": grouped.size(),
            "avg_seats": (totals["SEATS"] / grouped.size()).round(0).astype(int),
        }
    )
    for tier, column in TIER_COLUMNS.items():
        table[tier + "_pct"] = (100 * totals[column] / totals["SEATS"]).round(1)
        table[tier + "_pax_per_flight"] = (totals[column] / grouped.size()).round(1)
    return table.sort_values("flights", ascending=False).reset_index()


def compare_groupings(schedule):
    """Test each grouping the brief suggests to see which one changes the result."""
    parts = []
    for name, column in [
        ("Time of day", "TIME_OF_DAY"),
        ("Haul", "HAUL"),
        ("Region", "ARRIVAL_REGION"),
    ]:
        table = eligibility_table(schedule, [column]).rename(columns={column: "group"})
        table.insert(0, "grouping", name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def apply_to_flights(schedule, lookup, flight_numbers):
    """Estimate eligible passengers per flight: group % x seats on the flight."""
    flights = schedule[schedule["FLIGHT_NO"].isin(flight_numbers)].merge(
        lookup[["CATEGORY", "tier1_pct", "tier2_pct", "tier3_pct"]], on="CATEGORY"
    )
    for tier in TIER_COLUMNS:
        flights[tier + "_est_pax"] = (flights[tier + "_pct"] * flights["SEATS"] / 100).round(1)
    columns = ["FLIGHT_NO", "FLIGHT_DATE", "FLIGHT_TIME", "ARRIVAL_STATION_CD", "CATEGORY", "SEATS"]
    columns += [tier + "_est_pax" for tier in TIER_COLUMNS]
    return flights[columns]


def daily_demand(schedule):
    days = schedule["FLIGHT_DATE"].nunique()
    summary = {"flights_per_day": round(len(schedule) / days, 1)}
    for tier, column in TIER_COLUMNS.items():
        summary[tier + "_pax_per_day"] = round(schedule[column].sum() / days, 0)
    return pd.DataFrame([summary])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    schedule = load_schedule()

    outputs = {
        "grouping_comparison.csv": compare_groupings(schedule),
        "lounge_eligibility_lookup.csv": eligibility_table(schedule, ["CATEGORY"]),
        "daily_demand.csv": daily_demand(schedule),
    }
    outputs["sample_flights.csv"] = apply_to_flights(
        schedule, outputs["lounge_eligibility_lookup.csv"], SAMPLE_FLIGHTS
    )
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
        print("\n== {} ==".format(name))
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
