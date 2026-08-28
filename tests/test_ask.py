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
