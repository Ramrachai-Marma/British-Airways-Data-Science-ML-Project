# Task 1 (Data Analysis): Modeling lounge eligibility at Heathrow Terminal 3

## Problem statement

British Airways needs to forecast lounge demand at Heathrow Terminal 3. Future schedules and fleet plans are uncertain, so the Airport Planning team wants a way to estimate how many passengers will be eligible for each lounge without knowing exact flight numbers or aircraft.

Lounge access depends on loyalty status and travel class:

| Tier | Lounge | Who is eligible |
| --- | --- | --- |
| 1 | Concorde Room | First Class customers, BA Premier cardholders, BA Gold Guest List members |
| 2 | First Lounge | BA Gold members |
| 3 | Club Lounge | BA Silver cardholders, Club World (business class) customers |

Terminal 3 has no Concorde Room today, so Tier 1 is treated as hypothetical demand that could inform a future lounge.

The full brief is in [docs/problem_statement.pdf](docs/problem_statement.pdf).

## What the task asked for

1. Group flights into broad, reusable categories, for example by time of day, route type or destination region.
2. Estimate the percentage of passengers eligible for Tier 1, Tier 2 and Tier 3 in each category.
3. Apply the percentages to a sample of flights from the schedule.
4. Submit a lookup table and a short justification covering the grouping, the reasoning, the assumptions and how the model scales to future schedules.

## Data

[data/ba_summer_schedule.xlsx](data/ba_summer_schedule.xlsx): 10,000 BA departures from Heathrow, April to October 2025. Each row has the flight time and time-of-day band, destination, region, haul, aircraft type, seats by cabin, and the number of passengers eligible for each tier.

## What I did

**1. Calculated eligibility instead of guessing it.** The dataset already counts eligible passengers per flight, so each percentage is:

> Eligibility % = eligible passengers ÷ total seats (first + business + economy), summed across the group

**2. Tested every grouping the brief suggests** to see which ones change the result.

| Grouping tested | Tier 3 % by group | Finding |
| --- | --- | --- |
| Time of day | 13.4% to 14.0% in every band | Almost no difference |
| Haul | 16.9% short-haul, 10.5% long-haul | Clear difference |
| Region | Europe 16.9%; North America, Asia and Middle East 10.3% to 10.5% | Europe differs because it is all short-haul; the long-haul regions are nearly identical |

Haul is the driver. I kept region as a second label because it maps easily onto any future schedule.

**3. Built the lookup table** by haul and region.

**4. Applied it to sample flights** by multiplying the category percentage by the seats on the flight.

All of this is done in [lounge_eligibility.py](lounge_eligibility.py), which writes its tables to [outputs/](outputs/).

## Results

### Lounge eligibility lookup table

| Grouping | Tier 1 % | Tier 2 % | Tier 3 % | Flights | Avg seats | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Short-haul - Europe | 0.3% | 4.4% | 16.9% | 5,975 | 180 | A320; highest share because the aircraft is small |
| Long-haul - North America | 0.2% | 2.8% | 10.5% | 2,658 | 292 | Largest long-haul group |
| Long-haul - Asia | 0.2% | 2.7% | 10.4% | 679 | 291 | |
| Long-haul - Middle East | 0.2% | 2.7% | 10.3% | 688 | 292 | |

### Applied to sample flights

| Flight | Time | Destination | Category | Seats | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BA1896 | 15:33 | Frankfurt | Short-haul - Europe | 180 | 0.5 | 7.9 | 30.4 |
| BA5211 | 14:19 | Los Angeles | Long-haul - North America | 235 | 0.5 | 6.6 | 24.7 |

Across the whole schedule (about 48 flights a day) this comes to roughly 29 Tier 1, 379 Tier 2 and 1,449 Tier 3 eligible passengers per day.

### Key findings

- **Eligible passengers per flight are nearly constant**: about 0.6, 8 and 30 for Tiers 1, 2 and 3, whatever the aircraft size. That is why larger long-haul aircraft show lower percentages.
- **Time of day does not change eligibility**, although it still matters for when the lounges are busy.
- **Tier 1 demand is very small**, about 29 passengers a day, which is relevant to whether a Concorde Room at Terminal 3 would be justified.
- **These percentages are much lower than the illustrative ones in the Forage example** (55% to 75% for Tier 3), because they are calculated from the dataset, not assumed.

## Justification

| Question | Answer |
| --- | --- |
| How did you group the flights? | By route type (short-haul or long-haul), split by destination region. |
| Why that method? | I tested time of day, haul and region against the data. Only haul changed the eligibility percentages, and region keeps the table easy to map onto any schedule. |
| What assumptions did you make? | Percentages are a share of seat capacity, so they assume full flights. Time of day does not affect eligibility. The summer pattern holds for future schedules. Tier 1 is potential demand only, since Terminal 3 has no Concorde Room. |
| How does it apply to future schedules? | A new flight only needs a haul and a region to get a percentage; no flight numbers or aircraft details are required. A new region can borrow the long-haul rate until real data exists. |

## Reproduce

From the repository root:

```bash
python -m pip install -r requirements.txt
python task1-data-analysis-lounge-eligibility/lounge_eligibility.py
```

| Output | Contents |
| --- | --- |
| [outputs/lounge_eligibility_lookup.csv](outputs/lounge_eligibility_lookup.csv) | The lookup table |
| [outputs/grouping_comparison.csv](outputs/grouping_comparison.csv) | Eligibility by time of day, haul and region |
| [outputs/sample_flights.csv](outputs/sample_flights.csv) | The lookup table applied to two flights |
| [outputs/daily_demand.csv](outputs/daily_demand.csv) | Average eligible passengers per day |
