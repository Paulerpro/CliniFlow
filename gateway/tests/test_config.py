import os

from gateway.config import load_dotenv, proxy_auth_headers


def test_dotenv_loads_only_gateway_keys_and_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nVLLM_URL=http://from-file:8000\nMAX_RETRIES = '5'\n"
        "OPENAI_API_KEY=secret\nS3_BUCKET_NAME=bucket\nALLOW_LENIENT_PARSE=true\n",
        encoding="utf-8")
    for k in ("VLLM_URL", "MAX_RETRIES", "OPENAI_API_KEY", "S3_BUCKET_NAME", "ALLOW_LENIENT_PARSE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALLOW_LENIENT_PARSE", "false")  # a real variable must beat the file

    load_dotenv(env)
    try:
        assert os.environ["VLLM_URL"] == "http://from-file:8000"
        assert os.environ["MAX_RETRIES"] == "5"
        assert os.environ["ALLOW_LENIENT_PARSE"] == "false"
        assert "OPENAI_API_KEY" not in os.environ and "S3_BUCKET_NAME" not in os.environ
    finally:
        monkeypatch.delenv("VLLM_URL", raising=False)
        monkeypatch.delenv("MAX_RETRIES", raising=False)


def test_missing_dotenv_is_fine(tmp_path):
    load_dotenv(tmp_path / "nope.env")


def test_proxy_auth_headers_need_both_values():
    assert proxy_auth_headers({}) == {}
    assert proxy_auth_headers({"MODAL_PROXY_TOKEN_ID": "wk-1"}) == {}
    assert proxy_auth_headers({"MODAL_PROXY_TOKEN_ID": "wk-1", "MODAL_PROXY_TOKEN_SECRET": "ws-2"}) == {
        "Modal-Key": "wk-1", "Modal-Secret": "ws-2"}


def test_dotenv_reads_proxy_tokens_but_not_the_modal_cli_credentials(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MODAL_PROXY_TOKEN_ID=wk-1\nMODAL_TOKEN_ID=cli-id\nMODAL_TOKEN_SECRET=cli-secret", encoding="utf-8")
    for k in ("MODAL_PROXY_TOKEN_ID", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(env)
    try:
        assert os.environ["MODAL_PROXY_TOKEN_ID"] == "wk-1"
        assert "MODAL_TOKEN_ID" not in os.environ and "MODAL_TOKEN_SECRET" not in os.environ
    finally:
        monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
