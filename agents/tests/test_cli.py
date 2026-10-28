import io

import pytest

from agents import cli, signoff

from .conftest import FLAG_WARNING, gateway_response


class FakeTty:
    def __init__(self, text, tty=True):
        self._lines, self._tty = io.StringIO(text), tty

    def isatty(self):
        return self._tty

    def readline(self):
        return self._lines.readline()


def run(argv, store, stdin=None):
    out = io.StringIO()
    cli.run(cli.build_parser().parse_args(argv), store, stdin=stdin, stdout=out)
    return out.getvalue()


@pytest.fixture
def queued(store):
    nid = store.create_draft("t", gateway_response(code_review=True, warnings=[FLAG_WARNING]))
    signoff.request_signoff(store, nid)
    return nid


def test_approve_refuses_without_a_terminal(store, queued):
    with pytest.raises(signoff.SignoffError, match="interactive terminal"):
        run(["approve", queued, "--reviewer", "Dr. Lee", "--reason", "Checked it carefully."],
            store, FakeTty(queued + "\n", tty=False))
    assert store.get(queued)["status"] == "awaiting_review"


def test_approve_needs_the_id_typed_back(store, queued):
    with pytest.raises(signoff.SignoffError, match="cancelled"):
        run(["approve", queued, "--reviewer", "Dr. Lee", "--reason", "Checked it carefully."],
            store, FakeTty("yes\n"))
    assert store.get(queued)["status"] == "awaiting_review"


def test_approve_with_confirmation(store, queued):
    out = run(["approve", queued, "--reviewer", "Dr. Lee", "--reason", "Checked it carefully."],
              store, FakeTty(queued + "\n"))
    assert f"approved {queued}" in out and store.get(queued)["status"] == "approved"


def test_flagged_approve_without_reason_is_refused_even_after_confirmation(store, queued):
    with pytest.raises(signoff.SignoffError, match="written reason"):
        run(["approve", queued, "--reviewer", "Dr. Lee"], store, FakeTty(queued + "\n"))


def test_reject_requires_a_reason_argument():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["reject", "abc", "--reviewer", "Dr. Lee"])


def test_list_show_and_audit(store, queued):
    assert queued in run(["list", "--status", "awaiting_review"], store)
    shown = run(["show", queued], store)
    assert "code_repaired" in shown and "N809" in shown
    assert "requested_signoff" in run(["audit", queued], store)
