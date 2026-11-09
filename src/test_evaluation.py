from evaluation import compute_accuracy, compute_macro_f1


def test_perfect_prediction():
    # every prediction matches the gold label, so accuracy and macro F1 should both come out to a perfect 1.0
    gold = ['anger', 'joy', 'guilt']
    pred = ['anger', 'joy', 'guilt']
    assert compute_accuracy(gold, pred) == 1.0
    assert compute_macro_f1(gold, pred, ['anger', 'joy', 'guilt']) == 1.0


def test_all_wrong():
    # every prediction is wrong, so accuracy and macro F1 should both come out to 0.0
    gold = ['anger', 'joy', 'guilt']
    pred = ['joy', 'guilt', 'anger']
    assert compute_accuracy(gold, pred) == 0.0
    assert compute_macro_f1(gold, pred, ['anger', 'joy', 'guilt']) == 0.0 


def test_partial_correct():
    # 3 out of 4 predictions match, so accuracy should be 0.75, only checking accuracy here, not macro F1
    gold = ['anger', 'joy', 'guilt', 'anger']
    pred = ['anger', 'joy', 'anger', 'anger']
    assert compute_accuracy(gold, pred) == 0.75


if __name__ == '__main__':
    test_perfect_prediction()
    test_all_wrong()
    test_partial_correct()
    print("All tests passed!")