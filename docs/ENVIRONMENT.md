# Environment setup

StructAgent runs against two environments: an **OSWorld** desktop VM and a
**Chrome** browser for Mind2Web. Both speak the same `DesktopEnv` interface.

## OSWorld desktop VM

The desktop environment is OSWorld's (`desktop_env/`). Pick a provider:

| Provider | `--provider_name` | Notes |
|---|---|---|
| Docker | `docker` | Easiest locally; needs a GPU-free x86 host with KVM. |
| VMware | `vmware` | The OSWorld reference setup. |
| AWS    | `aws`    | For large parallel runs (`--num_envs`). |

Follow the OSWorld setup guide for VM images and provider prerequisites:
<https://github.com/xlang-ai/OSWorld#-quick-start>. Once a provider is configured,
the runner boots/﻿resets VMs automatically per task.

> The default task split is `evaluation_examples/test_nogdrive.json` (360 tasks),
> which excludes the handful of tasks needing Google-Drive OAuth, so it runs without
> any Google account. Use `test_all.json` only if you set up Google credentials under
> `evaluation_examples/settings/googledrive/`.

## Mind2Web (browser)

Mind2Web tasks drive a Chrome instance inside the same VM via the Chrome DevTools
Protocol (port 9222). No extra setup beyond a working desktop VM with Chrome and
`playwright install chromium` on the host (used by the grader/utilities).

## Credentials

Templates live in `evaluation_examples/settings/`. Provide your own where needed:
- `googledrive/` — only for Drive tasks (`test_all.json`); add `client_secrets.json` + `credentials.json` (git-ignored).
- `thunderbird/`, `proxy/` — the shipped values are the public OSWorld test defaults.
