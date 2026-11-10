from gateway.icd10 import normalize_code, resolve_code


def test_table_size(table):
    assert len(table) == 98186


def test_normalize_code():
    assert normalize_code("z20.822") == "Z20822"


def test_lookup_ignores_dots_and_case(table):
    entry = table.lookup("n80.9")
    assert entry.code == "N809" and entry.billable


def test_correct_code_is_verified(table):
    r = resolve_code(table, "N809", "ENDOMETRIOSIS, UNSPECIFIED")
    assert r.status == "verified" and r.code == "N809"


def test_dotted_code_is_normalised(table):
    assert resolve_code(table, "N80.9", "Endometriosis, unspecified").code == "N809"


def test_parent_code_is_accepted_not_repaired(table):
    # N508 is a real parent code but not billable; the dataset's own gold notes use such codes.
    r = resolve_code(table, "N508", "OTHER SPECIFIED DISORDERS OF MALE GENITAL ORGANS")
    assert r.status == "verified_non_billable" and r.code == "N508"


def test_wrong_number_right_name_is_repaired(table):
    # The model's typical failure: right diagnosis name, wrong code (seen on test row 2).
    r = resolve_code(table, "N83", "ENDOMETRIOSIS, UNSPECIFIED")
    assert r.status == "repaired" and r.code == "N809" and r.original_code == "N83"


def test_nonexistent_code_with_good_description_is_repaired(table):
    r = resolve_code(table, "Q9999", "Endometriosis, unspecified")
    assert r.status == "repaired" and r.code == "N809"


def test_nonexistent_code_with_junk_description_is_unresolved(table):
    r = resolve_code(table, "Q9999", "zzz qqq")
    assert r.status == "unresolved_not_found" and r.code == "Q9999"


def test_real_code_wrong_description_with_no_match_is_kept_and_flagged(table):
    r = resolve_code(table, "N809", "zzz qqq")
    assert r.status == "unresolved_mismatch" and r.code == "N809"


def test_empty_description_is_not_repaired(table):
    assert resolve_code(table, "Q9999", "").status == "unresolved_not_found"


def test_tie_goes_to_code_nearest_the_models(table):
    match = table.match_description("Unspecified", near_code="N809")
    assert match is not None


def test_search_ranks_clinical_phrases_first(table):
    for query, expected in [("endometriosis unspecified", "N809"),
                            ("type 2 diabetes with diabetic chronic kidney disease", "E1122"),
                            ("ventral hernia with obstruction", "K436")]:
        assert table.search(query)[0][0].code == expected


def test_search_drops_junk_and_hides_non_billable_by_default(table):
    assert table.search("zzz qqq") == []
    assert table.search("") == []
    assert all(e.billable for e, _ in table.search("other specified disorders of male genital organs"))
