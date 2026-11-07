# Reproduction scope

The public source profile supports source installation, local synthetic examples
and public documentation. Follow [development](../DEVELOPMENT.md) for those
checks. Record the exact commit and dependency environment when reporting a bug
or reusing the software.

Historical research registrations, experiment queues, datasets, model weights,
outcomes and complete provenance bundles are retained separately. This archive
does not contain a complete research reproduction bundle. Reproduction of a
specific study requires its original input versions, access rights, frozen
configuration, seeds, commands, commits and complete success/failure accounting.
No benchmark or quality result is established by this source profile.

## Systems measurements

The [performance guide](performance.md) describes measurement boundaries rather
than publishing results. A CPU oracle run checks a reference behavior. A native
engine replay measures a narrower path than model generation. Neither establishes
end-to-end latency, throughput or answer quality. Retain setup and memory costs,
failures and completion status with any measurement.

## Controlled serving timing protocol

Use a dedicated, idle device and matched model, tokenizer, prompts, request order,
sampling, seeds, token limits, concurrency and cache conditions. Declare whether
startup, warm-up, queueing, model prefill, CPU/tool calls and final delivery are
inside the interval. Preserve failed attempts and every consumed call. Randomize
arm order over repeated trials and distinguish repeatability from question-sample
uncertainty. A claim-bearing study additionally needs its registered protocol and
complete retained provenance; this page is not a substitute for those records.
