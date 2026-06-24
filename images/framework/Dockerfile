FROM python:3.11-slim

ARG CODEX_VERSION=latest
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
        ripgrep \
        rustc && \
    rm -rf /var/lib/apt/lists/*

COPY ${FRAMEWORK_REQUIREMENTS} /tmp/framework-requirements.txt
RUN pip install --no-cache-dir -r /tmp/framework-requirements.txt

RUN npm config set registry https://registry.npmmirror.com && \
    npm install -g "@openai/codex@${CODEX_VERSION}" && \
    codex --version

WORKDIR /root
RUN mkdir -p /root/results /logs/verifier
