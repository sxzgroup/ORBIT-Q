FROM python:3.11-slim

ARG CODEX_VERSION=latest
ARG CLAUDE_CODE_VERSION=latest
ARG FORGECODE_VERSION=latest
ARG FRAMEWORK_REQUIREMENTS=frameworks/tensorcircuit/requirements.txt

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN for file in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
        if [ -f "$file" ]; then \
            sed -i \
                -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
                -e 's|https://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
                -e 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
                -e 's|https://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
                -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
                -e 's|https://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
                "$file"; \
        fi; \
    done && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cargo \
        curl \
        git \
        nodejs \
        npm \
        pkg-config \
        procps \
        ripgrep \
        rustc && \
    rm -rf /var/lib/apt/lists/*

COPY ${FRAMEWORK_REQUIREMENTS} /tmp/framework-requirements.txt
RUN pip install --no-cache-dir -r /tmp/framework-requirements.txt

COPY scripts/patch_torchquantum_optional_ibm_runtime.py /tmp/patch_torchquantum_optional_ibm_runtime.py
RUN python /tmp/patch_torchquantum_optional_ibm_runtime.py

RUN npm config set registry https://registry.npmmirror.com && \
    npm install -g "@openai/codex@${CODEX_VERSION}" && \
    codex --version

RUN npm config set registry https://registry.npmmirror.com && \
    npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" && \
    claude --version

RUN set -eux; \
    case "$(uname -m)" in \
        x86_64|amd64) forge_arch=x86_64 ;; \
        aarch64|arm64) forge_arch=aarch64 ;; \
        *) echo "Unsupported ForgeCode architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    if [ "${FORGECODE_VERSION}" = "latest" ]; then \
        forge_release_path="latest/download"; \
    else \
        case "${FORGECODE_VERSION}" in \
            v*) forge_tag="${FORGECODE_VERSION}" ;; \
            *) forge_tag="v${FORGECODE_VERSION}" ;; \
        esac; \
        forge_release_path="download/${forge_tag}"; \
    fi; \
    curl -fsSL --retry 3 \
        "https://github.com/tailcallhq/forgecode/releases/${forge_release_path}/forge-${forge_arch}-unknown-linux-gnu" \
        -o /usr/local/bin/forge; \
    chmod 0755 /usr/local/bin/forge; \
    forge --version

WORKDIR /root
RUN mkdir -p /root/results /logs/verifier
