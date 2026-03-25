# --- STAGE 1: BUILDER ---
FROM rust:slim-bookworm as builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . .

RUN cargo build --release

# --- STAGE 2: RUNTIME (Production) ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -U pytja_user

WORKDIR /app

COPY --from=builder /usr/src/app/target/release/pytja /usr/local/bin/pytja

RUN mkdir -p /app/data /app/logs && chown -R pytja_user:pytja_user /app

USER pytja_user

ENV RUST_LOG=info
EXPOSE 50051

CMD ["pytja", "server"]