"""30-query regression harness for real agent evaluation."""

from pipeline.run import process

QUERIES = [
    "Find active cardiology providers in Los Angeles hospitals that accept Blue Shield",
    "Show inactive oncology clinics in Seattle",
    "List providers accepting Cigna Choice plan in Texas",
    "Find providers with open appointments in Austin urgent care centers",
    "Show pediatric specialists in San Diego accepting Blue Shield PPO",
    "Show orthopedic surgeons in Chicago with hospital affiliations",
    "Find dermatology providers in Miami who take United HMO",
    "Get rheumatology specialists in Denver clinics",
    "Show active providers in Phoenix with appointments this week",
    "Find cardiologists near Dallas urgent care centers",
    "List nephrology doctors in Boston accepting Medicare Advantage",
    "Find neurologists in San Francisco hospitals",
    "Show family medicine providers in Portland taking Cigna Choice",
    "Find active endocrinologists in Atlanta facilities",
    "Show orthopedic providers in Minneapolis accepting Blue Shield PPO",
    "Find oncology specialists in Sacramento clinics",
    "Show pediatric cardiologists in Houston hospitals",
    "Find pulmonologists in Tampa accepting Aetna Select",
    "Show infectious disease doctors in Seattle hospitals",
    "Find gastroenterologists in New York accepting Empire PPO",
    "Show ophthalmologists in Las Vegas clinics",
    "Find ENT specialists in Philadelphia hospitals",
    "Show allergy specialists in Charlotte taking Blue Shield PPO",
    "Find urologists in San Jose accepting Cigna Choice",
    "Show hematologists in Detroit hospitals",
    "Find geriatricians in Orlando clinics",
    "Show sports medicine providers in Denver taking Kaiser HMO",
    "Find podiatrists in Phoenix accepting Medicare Advantage",
    "Show nurse practitioners in Austin clinics",
    "Find cardiology providers in Los Angeles accepting Anthem Gold",
]


def main() -> int:
    results = []
    for idx, text in enumerate(QUERIES, start=1):
        try:
            res = process(text)
            results.append((idx, text, True, res["query"].strip()))
        except Exception as exc:  # pragma: no cover - dev-only harness
            results.append((idx, text, False, str(exc)))

    score = sum(1 for _, _, ok, _ in results if ok)
    print(f"SCORE {score}/{len(QUERIES)}")

    for idx, text, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"[{idx:02d}] {status} :: {text}")
        snippet = detail.replace("\n", " ")[:200]
        prefix = "GraphQL" if ok else "Error"
        print(f"    {prefix}: {snippet}...")

    return 0 if score == len(QUERIES) else 2

if __name__ == "__main__":
    raise SystemExit(main())
