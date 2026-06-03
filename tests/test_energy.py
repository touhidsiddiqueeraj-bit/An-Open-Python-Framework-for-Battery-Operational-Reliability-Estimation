"""Unit tests for energy unit conversion."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

def test_energy_unit_conversion():
    """kWh × $/MWh / 1000 = correct revenue. Without /1000, 1000× overstatement."""
    # Direct revenue calculation test (simulates MarketSimulator logic)
    energy_kwh = 150.0 * 0.5  # 150 cycles × 0.5 kWh = 75 kWh
    price_per_mwh = 50.0

    # Correct: divide by 1000 (kWh → MWh)
    correct = energy_kwh * price_per_mwh / 1000.0
    # Wrong: omit division
    wrong = energy_kwh * price_per_mwh

    assert abs(correct - 3.75) < 0.01, f"Should be $3.75, got {correct}"
    assert abs(wrong - 3750.0) < 1.0, f"Wrong should be $3,750, got {wrong}"
    assert wrong / correct > 999, f"Ratio should be ~1000, got {wrong/correct:.0f}"
    print(f"PASS: correct=${correct:.2f}, wrong=${wrong:.0f}, ratio={wrong/correct:.0f}x")


def test_market_simulator_revenue_scale():
    """Verify MarketSimulator's pricing logic uses correct units."""
    from src.dispatch.market_sim import MarketSimulator
    from src.dispatch.threshold import ThresholdPolicy
    import numpy as np

    sim = MarketSimulator(
        price_mean=50.0, price_std=0.0, price_ar_coeff=0.0,
        service_energy_kwh=0.5, penalty_cost=500.0, seed=42)

    policy = ThresholdPolicy(tau=1.0)  # always dispatch (τ=1 accepts all)
    P_cal = np.full(150, 0.1)  # 150 cycles, 10% failure probability
    result = sim.run(P_cal, eol_cycle=1000, horizon=10, dispatch_policy=policy)

    # With deterministic $50/MWh price and 0.5 kWh per cycle for 150 cycles:
    # revenue = 150 × 0.5 × 50 / 1000 = $3.75
    expected = 3.75
    assert abs(result["revenue"] - expected) < expected * 0.5, (
        f"Revenue {result['revenue']:.2f} far from expected {expected:.2f}")
    print(f"PASS: MarketSimulator revenue=${result['revenue']:.2f} "
          f"(expected ~${expected:.2f})")


if __name__ == "__main__":
    test_energy_unit_conversion()
    test_market_simulator_revenue_scale()
