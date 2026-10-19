# Serving image for CliniFlow. Same image runs locally (docker run --gpus all)
# and on the AWS GPU instance. The adapter is NOT baked in: it is mounted or
# downloaded at start, so one image serves any adapter version.
FROM vllm/vllm-openai:latest

WORKDIR /serving
COPY config.py start_server.sh client.py ./
RUN chmod +x start_server.sh

# Override the image's default entrypoint so our script controls the flags.
ENTRYPOINT ["/serving/start_server.sh"]
