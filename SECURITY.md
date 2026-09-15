# Security Notes

## Local and deployed defaults

BrainOS is local-first. For any network-exposed deployment:

- keep local mode on localhost, or set `BRAINOS_AUTH_MODE=multi_user` for SQL-backed registration/login and ownership checks;
- for simple single-operator deployments, a strong `BRAINOS_AUTH_TOKEN` remains supported;
- set `BRAINOS_CORS_ORIGINS` to exact trusted browser origins;
- terminate TLS at a reverse proxy and forward WebSocket upgrades to `/ws`;
- keep provider keys and auth tokens in the deployment secret manager, never in the frontend bundle, image, Git, or replay files;
- persist the replay directory, model cache, and relational database volume that are required;
- expose `/api/health` for process health and `/api/ready` for model readiness.

## Data boundaries

Prompts, generated output, event streams, and replay captures can contain sensitive user data. Restrict filesystem permissions on `BRAINOS_REPLAY_DIR`, model-cache, and database volumes. In multi-user mode, session/run/replay REST and WebSocket access is checked against the authenticated owner. Legacy replay files are claimable by the first authenticated user who imports them; do not copy private replay files between users.

External provider mode is observation-only. BrainOS does not fabricate or infer private provider tensors such as hidden states, attention, Q/K/V, logits, or MLP activations.

## Current limitations

Multi-user authentication is implemented with Argon2 password hashes, opaque database-backed sessions, SQLite development storage, and PostgreSQL-compatible SQLAlchemy URLs. It still requires deployment-specific TLS, secret management, backup, rate-limit, and operational review before being treated as a production multi-tenant service. CUDA, cloud deployment, and third-party provider credentials require separate environment/hardware verification.

Report suspected security issues privately to the repository owner rather than including live credentials or replay artifacts in an issue.
