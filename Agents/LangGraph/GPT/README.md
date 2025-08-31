## Prerequis
```sh
# Prepare shell tool user & dir
sudo adduser --system --shell /bin/bash --group --no-create-home agentexecutor
sudo mkdir -p /safe/agent/workdir
sudo chown agentexecutor: /safe/agent/workdir
sudo chmod 700 /safe/agent/workdir
# Add one line in sudoer
# ++ user_exec_script ALL=(agentexecutor) NOPASSWD: ALL

# Prepare search tool API token
export TAVILY_API_KEY=tvly-foobar

# Underlying LLM
ollama run gpt-oss:20b
```
