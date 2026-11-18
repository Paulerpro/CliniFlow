from gateway.grounding import check_grounding

TRANSCRIPT = (
    "[doctor] Your blood pressure is 120/80 and heart rate is 76 bpm. "
    "I will start Lisinopril 10 mg daily and see you in three months."
)


def note(**kw):
    base = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    base.update(kw)
    return base


def types(warnings):
    return {w.type for w in warnings}


def test_grounded_note_has_no_warnings():
    n = note(objective="BP 120/80, HR 76 bpm.", plan="Lisinopril 10 mg daily. Return in 3 months.")
    assert check_grounding(TRANSCRIPT, n) == []


def test_invented_vital_is_flagged_for_review():
    ws = check_grounding(TRANSCRIPT, note(objective="Temperature 101.4°F, SpO2 91%."))
    w = next(w for w in ws if w.type == "ungrounded_measured_value")
    assert w.severity == "review" and "101.4" in w.detail


def test_invented_medication_is_flagged():
    ws = check_grounding(TRANSCRIPT, note(plan="Start Metformin 500 mg twice daily."))
    assert "ungrounded_medication" in types(ws)


def test_right_drug_wrong_dose_is_flagged():
    ws = check_grounding(TRANSCRIPT, note(plan="Lisinopril 40 mg daily."))
    assert "ungrounded_medication" in types(ws)


def test_invented_doctor_is_flagged():
    ws = check_grounding(TRANSCRIPT, note(plan="Refer to Dr. Emily Carter."))
    assert "ungrounded_person" in types(ws)


def test_spelled_out_number_counts_as_grounded():
    assert check_grounding(TRANSCRIPT, note(plan="Follow up in 3 months.")) == []


def test_invented_demographics_are_info_only():
    ws = check_grounding(TRANSCRIPT, note(subjective="A 45-year-old African American male, Mr. Smith."))
    w = next(w for w in ws if w.type == "invented_demographic")
    assert w.severity == "info"
    assert "45-year-old" in w.detail and "African American" in w.detail


def test_demographics_stated_in_transcript_are_fine():
    t = TRANSCRIPT + " You are 45 years old."  # the age number is now in the transcript
    ws = check_grounding(t, note(subjective="A 45-year-old patient."))
    assert "invented_demographic" not in types(ws)


def test_leading_verb_is_not_mistaken_for_the_drug():
    # "Start" / "Prescribed" are verbs: the drug is the word before the dose, and it is in the transcript
    t = TRANSCRIPT + " We will also use montelukast 10 mg at night."
    for sentence in ("Start montelukast 10 mg nightly.", "Prescribed montelukast 10 mg at bedtime."):
        assert "ungrounded_medication" not in types(check_grounding(t, note(plan=sentence)))


def test_lowercase_invented_drug_after_a_verb_is_still_caught():
    ws = check_grounding(TRANSCRIPT, note(plan="Start warfarin 5 mg daily."))
    w = next(w for w in ws if w.type == "ungrounded_medication")
    assert "warfarin" in w.detail and "Start" not in w.detail


def test_amounts_that_are_not_drugs_are_ignored():
    ws = check_grounding(TRANSCRIPT, note(plan="Maximum 4000 mg per day. Dose 250 mg."))
    assert "ungrounded_medication" not in types(ws)
