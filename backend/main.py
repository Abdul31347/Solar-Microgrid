
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import math

app = FastAPI(title="Solar Microgrid Scheduler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

H = 24
MU = 30
Q = 10
MASK = 0xFFFFFFFF


class SchedulerRequest(BaseModel):
    average_load: float = Field(default=4, gt=0, le=20)
    solar_peak: float = Field(default=10, ge=0, le=30)
    battery_capacity: float = Field(default=20, ge=2, le=100)
    diesel_price: float = Field(default=100, ge=10, le=300)
    generations: int = Field(default=300, ge=20, le=1000)
    seed: int = 42


# Reproducible random numbers, following the frontend's generator.
class RandomGenerator:
    def __init__(self, seed):
        self.seed = int(seed) & MASK

    def random(self):
        self.seed = (self.seed + 0x6D2B79F5) & MASK
        t = ((self.seed ^ (self.seed >> 15)) * (1 | self.seed)) & MASK
        value = ((t ^ (t >> 7)) * (61 | t)) & MASK
        t = (((t + value) & MASK) ^ t) & MASK
        t = (t ^ (t >> 14)) & MASK
        return t / 4294967296

    def gaussian(self):
        u = 0
        while u == 0:
            u = self.random()
        return math.sqrt(-2 * math.log(u)) * math.cos(
            2 * math.pi * self.random()
        )


# Create the 24-hour solar and load profiles.
def create_scenario(req):
    load = []
    pv = []

    for t in range(H):
        factor = (
            0.7
            + (0.3 if 7 <= t <= 19 else 0)
            + (0.3 if 18 <= t <= 22 else 0)
        )
        load.append(req.average_load * factor)

        solar = (
            req.solar_peak
            * math.sin(math.pi * (t - 6) / 12) ** 1.5
            if 6 <= t <= 18 else 0
        )
        pv.append(solar)

    return {
        "load": load,
        "pv": pv,
        "cap": req.battery_capacity,
        "price": req.diesel_price,
        "dmax": max(load) * 1.2,
        "bmax": req.battery_capacity * 0.3,
    }


# Simulate one complete daily schedule and calculate its fitness.
def simulate(x, s):
    soc = s["cap"] * 0.5
    starting_soc = soc
    fuel = 0
    unmet = 0
    violations = 0
    socs = [soc]

    for t in range(H):
        diesel = 0 if x[t] < 0.3 else x[t]
        battery = x[H + t]

        if diesel > 0:
            fuel += 0.27 * diesel + 0.6

        supply = s["pv"][t] + diesel + battery
        if supply < s["load"][t]:
            unmet += s["load"][t] - supply

        if battery < 0:
            soc += -battery * 0.92
        else:
            soc -= battery / 0.92

        if soc > s["cap"]:
            violations += soc - s["cap"]
            soc = s["cap"]

        if soc < s["cap"] * 0.2:
            violations += s["cap"] * 0.2 - soc
            soc = s["cap"] * 0.2

        socs.append(soc)

    ending_shortfall = max(0, starting_soc - soc)
    cost = fuel * s["price"]

    total = (
        cost
        + 1000 * unmet
        + 300 * violations
        + 300 * ending_shortfall
    )

    return {
        "fuel": fuel,
        "cost": cost,
        "unmet": unmet,
        "viol": violations,
        "es": ending_shortfall,
        "socs": socs,
        "total": total,
    }


# Create an initial candidate schedule.
def new_individual(s, rng):
    x = [
        rng.random() * s["dmax"] if rng.random() < 0.4 else 0
        for _ in range(H)
    ]

    x += [
        (rng.random() * 2 - 1) * s["bmax"]
        for _ in range(H)
    ]

    steps = [
        0.15 * (s["dmax"] if i < H else 2 * s["bmax"])
        for i in range(2 * H)
    ]

    return {"x": x, "steps": steps, "fitness": simulate(x, s)["total"]}


# Create a child using self-adaptive mutation.
def create_child(parent, s, rng):
    n = 2 * H
    tau = 1 / math.sqrt(2 * n)
    tau_prime = 1 / math.sqrt(2 * math.sqrt(n))
    shared_gaussian = rng.gaussian()

    x = []
    steps = []

    for i in range(n):
        exponent = tau * shared_gaussian + tau_prime * rng.gaussian()
        step = max(0.02, parent["steps"][i] * math.exp(exponent))

        low = 0 if i < H else -s["bmax"]
        high = s["dmax"] if i < H else s["bmax"]

        value = parent["x"][i] + step * rng.gaussian()
        value = max(low, min(high, value))

        x.append(value)
        steps.append(step)

    return {"x": x, "steps": steps, "fitness": simulate(x, s)["total"]}


# Tournament competition selects the next generation.
def next_generation(population, s, rng):
    children = [
        create_child(parent, s, rng)
        for parent in population
    ]
    candidates = population + children

    for candidate in candidates:
        wins = 0
        for _ in range(Q):
            rival = candidates[int(rng.random() * len(candidates))]
            if candidate["fitness"] <= rival["fitness"]:
                wins += 1
        candidate["wins"] = wins

    candidates.sort(key=lambda item: (-item["wins"], item["fitness"]))
    return candidates[:MU]


# Simple comparison: solar first, then battery, then diesel.
def create_baseline(s):
    x = [0.0] * (2 * H)
    soc = s["cap"] * 0.5

    for t in range(H):
        net = s["load"][t] - s["pv"][t]

        if net <= 0:
            charge = min(
                -net, s["bmax"], (s["cap"] - soc) / 0.92
            )
            x[H + t] = -charge
            soc += charge * 0.92
        else:
            discharge = min(
                net, s["bmax"], (soc - s["cap"] * 0.2) * 0.92
            )
            x[H + t] = discharge
            soc -= discharge / 0.92

            remaining = net - discharge
            if remaining > 0.001:
                x[t] = min(max(remaining, 0.3), s["dmax"])

    return x


# Run the complete Evolutionary Programming algorithm.
def optimize(req, seed):
    s = create_scenario(req)
    rng = RandomGenerator(seed)
    baseline_x = create_baseline(s)

    population = [
        new_individual(s, rng)
        for _ in range(MU)
    ]

    history = []

    for _ in range(req.generations):
        population = next_generation(population, s, rng)
        history.append(population[0]["fitness"])

    best = population[0]
    ep_result = simulate(best["x"], s)
    baseline_result = simulate(baseline_x, s)

    return {
        "scenario": s,
        "schedule": best["x"],
        "ep": ep_result,
        "baseline": baseline_result,
        "history": history,
        "generations": req.generations,
    }


def statistics(values):
    mean = sum(values) / len(values)
    std_dev = math.sqrt(
        sum((v - mean) ** 2 for v in values) / len(values)
    )

    return {
        "best": min(values),
        "mean": mean,
        "std_dev": std_dev,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Backend is running"}


@app.post("/api/scheduler/run")
def run_scheduler(req: SchedulerRequest):
    return optimize(req, req.seed)


@app.post("/api/scheduler/multi-run")
def run_multiple(req: SchedulerRequest):
    total_costs = []
    fuel_costs = []

    for i in range(1, 11):
        result = optimize(req, i * 7919)
        total_costs.append(result["ep"]["total"])
        fuel_costs.append(result["ep"]["cost"])

    return {
        "runs": 10,
        "total_cost": statistics(total_costs),
        "fuel_cost": statistics(fuel_costs),
    }
