def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


if __name__ == "__main__":
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
