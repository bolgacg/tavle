from tavle import ask


def test_guard_rejects_everything_that_is_not_one_select():
    assert ask.guard("CANNOT_ANSWER") == (False, "declined")
    assert ask.guard("delete from prices") [0] is False
    assert ask.guard("select 1; select 2")[0] is False
    assert ask.guard("select * from ops.runs")[0] is False
    assert ask.guard("with x as (select 1) select * from x join prices on 1=1")[0] is True
    assert ask.guard("select avg(price_eur) from power_hourly")[0] is True


def test_same_ignores_row_order():
    assert ask.same([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert not ask.same([(1,)], [(2,)])


def test_extra_columns_are_not_a_wrong_answer():
    # "which day was highest" answered with the day AND the price
    assert ask.contains([("2025-01-20",)], [("2025-01-20", 231.39)])
    # a float within tolerance still counts
    assert ask.contains([(7.47,)], [("2026-01-02", "DKK", "EUR", 7.4701)])
    # a different value does not
    assert not ask.contains([(7.47,)], [("2026-01-02", "DKK", "EUR", 7.61)])
    # row counts must still match
    assert not ask.contains([(1,)], [(1,), (2,)])


def test_close_only_applies_to_single_numbers():
    assert ask.close([(100.0,)], [(100.4,)])
    assert not ask.close([(100.0,)], [(110.0,)])
    assert not ask.close([(1.0,), (2.0,)], [(1.0,), (2.0,)])
