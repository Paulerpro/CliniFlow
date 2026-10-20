"""Send one transcript to the vLLM server and check the reply.

Standard library only, so it runs anywhere (laptop, Colab, EC2) with no installs.

Usage:
    python serving/client.py --transcript-file path/to/transcript.txt
    python serving/client.py --transcript "[doctor] ... [patient] ..."
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"]


def load_system_prompt(adapter_dir=config.ADAPTER_DIR):
    """Read the exact system prompt saved with the adapter at training time."""
    path = os.path.join(adapter_dir, "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_request(transcript, system_prompt):
    return {
        "model": config.ADAPTER_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": 1500,
        "temperature": 0,  # deterministic, so runs are comparable
    }


def generate(transcript, system_prompt, server_url=config.SERVER_URL, timeout=600):
    body = json.dumps(build_request(transcript, system_prompt)).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()


def check_soap(text):
    """Return (ok, problems). A reply passes if it is JSON with the SOAP fields
    and a non-empty suggested_codes list. Same shape as the Pillar 2 schema."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False, ["not_json"]
    problems = [f"missing_{f}" for f in SOAP_FIELDS if not isinstance(obj.get(f), str)]
    codes = obj.get("suggested_codes")
    if not isinstance(codes, list) or not codes:
        problems.append("suggested_codes_missing_or_empty")
    return not problems, problems


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--transcript")
    group.add_argument("--transcript-file")
    args = parser.parse_args()

    transcript = args.transcript
    if args.transcript_file:
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()

    reply = generate(transcript, load_system_prompt())
    print(reply)
    ok, problems = check_soap(reply)
    print("\nSCHEMA:", "PASS" if ok else f"FAIL {problems}")


if __name__ == "__main__":
    main()
