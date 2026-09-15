# Deployment

## Provider-neutral production shape

The repository does not select or claim a cloud provider. A production installation should keep the following boundary explicit:

```text
Browser
  ↓ HTTPS / TLS reverse proxy
Frontend assets + API proxy + WebSocket upgrade
  ↓ private network
BrainOS API process
  ↓ serialized local inference runtime
Inference model runtime
  ↓
Persistent replay store + relational metadata (when multi-user is enabled)
  ↓
Persistent Hugging Face model cache
```

The reverse proxy must enforce HTTPS, forward `/api/*` and `/ws`, preserve WebSocket upgrades, and expose only the public frontend/API origin. The BrainOS process should run with a restart policy, structured request logs, `/api/health` liveness, `/api/ready` model readiness, and persistent model-cache/replay/database volumes. Provider secrets belong in the host secret manager or injected environment, never in the image or frontend bundle.

For multi-user mode, set `BRAINOS_AUTH_MODE=multi_user` and `BRAINOS_DATABASE_URL` to a persistent SQLite URL for development or a PostgreSQL URL such as `postgresql+psycopg://user:password@db/brainos` in production. The schema is managed by Alembic. Run `alembic upgrade head` during deployment, or leave `BRAINOS_DATABASE_AUTO_MIGRATE=true` for idempotent forward startup migrations; set it to `false` when the deployment process owns migration timing. Never use a destructive downgrade against production data.

## Docker Compose

The supported deployment artifact is the root `Dockerfile` and `docker-compose.yml`. The GitHub-hosted Linux CI runtime smoke builds and starts the image, verifies health/readiness, frontend serving, WebSocket connectivity, named-volume persistence across restart, and clean shutdown. Docker runtime remains unverified on the current Mac because Docker is not installed there.

1. Copy `.env.example` to `.env`.
2. Set `BRAINOS_AUTH_TOKEN` to a strong random value.
3. Set `BRAINOS_CORS_ORIGINS` to the public browser origin.
4. Build and start:

```bash
docker compose build
docker compose up -d
```

5. Follow model startup:

```bash
docker compose logs -f brainos
```

6. Open `http://localhost:8765` or the configured reverse-proxy URL.

The model cache and replay tensor bundles are stored in named volumes. The first container start downloads the selected Hugging Face model. CPU inference is supported in the container profile. The current verified MPS runtime is the native Mac path; the generic image is CPU-oriented and does not claim Apple MPS passthrough. A separate CUDA image/profile would require suitable NVIDIA hardware and explicit verification.

## Reverse Proxy

The example Nginx configuration is in `deploy/nginx/brainos.conf.example`. Replace the example hostname and certificate paths, then validate it with `nginx -t`. The proxy must forward both HTTP and WebSocket traffic to port `8765`, preserve the upgrade headers, and enforce HTTPS. If the frontend and backend share the same public origin, leave `VITE_BRAINOS_WS_URL` unset and the browser uses the current origin's `/ws` path. For a separate backend origin, set `VITE_BRAINOS_WS_URL` at frontend build time. GitHub CI validates the configuration syntax and runs an ephemeral TLS/WebSocket proxy smoke test; this does not verify a public production certificate or network.

## Secrets

Keep `BRAINOS_AUTH_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and provider keys in the deployment secret manager. They are read only by the backend. Never put them in the frontend bundle, Dockerfile, Git, or `.env.example`.

## Health, readiness, and operations

Use `/api/health` for process/container liveness and `/api/ready` for model readiness. A reverse proxy/load balancer must wait for readiness before routing inference traffic. `/api/hardware`, `/api/monitoring/history`, and the low-cardinality Prometheus-compatible `/api/metrics` endpoint expose runtime diagnostics; the API middleware protects metrics in authenticated modes. Logs include application errors and session identifiers where available, but production deployments should add structured collection, retention, redaction, and alerting policy.

## CI

`.github/workflows/ci.yml` runs backend tests, frontend typecheck/build, dependency audit, PostgreSQL migration/runtime and backup/restore smoke tests, Docker image/runtime validation, Nginx syntax/TLS proxy validation, and the real Playwright suite on GitHub-hosted Linux runners. The same browser suite is also reproducibly runnable on the development Mac.

## Hosting

No cloud host is configured by this repository. Docker Compose is the reproducible deployment target; a VPS, managed container service, or private server can run the same image with persistent volumes and a WebSocket-capable reverse proxy.
