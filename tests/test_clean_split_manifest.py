from reports.build_clean_split_manifest import UnionFind, choose_validation_groups


def test_union_find_connects_transitive_groups():
    groups = UnionFind(["a", "b", "c"])
    groups.union("a", "b")
    groups.union("b", "c")

    assert groups.find("a") == groups.find("c")


def test_validation_group_selection_keeps_groups_whole():
    groups = [["a", "b"], ["c"], ["d", "e", "f"], ["g"]]
    selected = choose_validation_groups(groups, target_images=3, seed=42)

    assert sum(len(groups[index]) for index in selected) == 3
