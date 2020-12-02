from common import EarlyStopping


def test_early_stopping_should_stop():
    patience = 2
    early_stopping = EarlyStopping(patience)

    assert not early_stopping.should_break
    assert early_stopping.is_best(1)
    early_stopping.update(1)

    # After 2 worse updates, should_break should return True
    assert not early_stopping.should_break
    assert not early_stopping.is_best(3)
    early_stopping.update(3)

    assert not early_stopping.should_break
    assert not early_stopping.is_best(2)
    early_stopping.update(2)

    assert early_stopping.should_break

    # But after updating with smaller value than the best one, it should reset
    assert early_stopping.is_best(0)
    early_stopping.update(0)
    assert not early_stopping.should_break


def test_early_stopping_disable():
    patience = 2
    early_stopping = EarlyStopping(patience)

    early_stopping.update(1)
    early_stopping.update(2)
    early_stopping.update(3)

    assert early_stopping.should_break

    early_stopping.disable()

    assert not early_stopping.should_break
