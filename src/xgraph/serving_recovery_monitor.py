"""Read-only operational inventory; scientific qualification belongs to the coordinator.

Optional controller-owned archives preserve each canonical frame plus LF. The physical
format is MAGIC, >I header length, canonical JSON header, then FRAM/>QII(sequence,
gzip length, plaintext length)/gzip-member records, and DONE/>I length/canonical
JSON anchors. All physical bytes are hashed. Gzip level/mtime/name/header are fixed;
member CRC, every logical anchor and exact EOF must verify before qualification.
The canonical current-state proxy is distinct from disk bytes and physical RSS.
"""

import copy
import csv
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import time
import zlib
from collections import deque
from pathlib import Path
from subprocess import TimeoutExpired

GPU_QUERY = ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
COMPUTE_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name",
    "--format=csv,noheader,nounits",
)
DOCKER_FORMAT = (
    '{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},'
    '"Config":{"Image":{{json .Config.Image}},"Labels":{{json .Config.Labels}}},'
    '"State":{"Running":{{json .State.Running}},"Pid":{{json .State.Pid}}}}'
)
OS_IDENTITY_DECISION = "a84e961d6c737e54e28388f6d8c42220d35590cbbbc1a75b197e072a1f39319d"
OS_CONTEXT_DECISION = "40be3505b14c4f2a2a4bfef481a47795597dc936f32ab3580b843bdc992a014b"
STORAGE_DECISION = "b6c984996e10a6663e88e26e6d8cdbb2b64fd2b31d3c0d22c76aa7c284342b64"
STORAGE_AMENDMENT_DECISION = "a2a817b4468c697d6215be480a0388bc802008f55eccde5c7dc39f5343e7a182"
STORAGE_CEILINGS = {
    "state_proxy_bytes": 67108864,
    "frame_plaintext_bytes": 8388608,
    "physical_archive_bytes": 1073741824,
    "canonical_archive_bytes": 17179869184,
    "frames": 65536,
}
ARCHIVE_FORMAT = "xgraph-monitor-gzip-members-v1"
ARCHIVE_NAME = "frames.xgm.gz"
ARCHIVE_MAGIC = b"XGMON8\0\n"
GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\xff"
ARCHIVE_JSON_LIMIT = 65536
SYSTEMD_PREFIX = (
    "/usr/bin/systemctl",
    "--system",
    "--no-pager",
    "show",
    "--property=MainPID,ControlGroup,FragmentPath,ExecStart",
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


class ArchiveError(ValueError):
    """A failed archive cannot qualify a trial or authorize an automatic retry."""


def _storage_limits(limits, bindings=None):
    ceilings = STORAGE_CEILINGS
    if (
        isinstance(bindings, dict)
        and isinstance(bindings.get("decisions"), dict)
        and bindings["decisions"].get("storage_amendment") == STORAGE_AMENDMENT_DECISION
    ):
        ceilings = {**ceilings, "physical_archive_bytes": 2147483648}
    if (
        not isinstance(limits, dict)
        or limits.keys() != ceilings.keys()
        or any(
            type(value) is not int or not 0 < value <= ceilings[key]
            for key, value in limits.items()
        )
    ):
        raise ArchiveError("explicit positive archive limits must fit the adopted ceilings")
    return copy.deepcopy(limits)


def _storage_bindings(bindings):
    if (
        not isinstance(bindings, dict)
        or set(bindings)
        != {
            "execution_id",
            "trial_id",
            "source_sha256",
            "policy_sha256",
            "intent_sha256",
            "decisions",
        }
        or any(
            not isinstance(bindings[key], str) or not 0 < len(bindings[key]) <= 4096
            for key in ("execution_id", "trial_id")
        )
        or any(
            not isinstance(bindings[key], str)
            or re.fullmatch("[0-9a-f]{64}", bindings[key]) is None
            for key in ("source_sha256", "policy_sha256", "intent_sha256")
        )
        or not isinstance(bindings["decisions"], dict)
        or bindings["decisions"].get("storage") != STORAGE_DECISION
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or re.fullmatch("[0-9a-f]{64}", value) is None
            for key, value in bindings["decisions"].items()
        )
    ):
        raise ArchiveError(
            "archive requires exact execution/trial/source/policy/intent/decision bindings"
        )
    return copy.deepcopy(bindings)


def _policy_storage_bindings(policy, intent):
    return _storage_bindings(
        {
            "execution_id": policy.get("execution_id"),
            "trial_id": intent.get("job_id"),
            "source_sha256": policy.get("source_sha256"),
            "policy_sha256": hashlib.sha256(
                json.dumps(policy, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "intent_sha256": _digest(intent),
            "decisions": policy.get("decisions"),
        }
    )


def _archive_directory(directory):
    path = Path(directory)
    if not path.is_absolute() or str(path.resolve(strict=True)) != str(path) or not path.is_dir():
        raise ArchiveError("archive directory must already exist at its exact canonical path")
    return path


def _archive_header(identity, limits, bindings):
    return {
        "format": ARCHIVE_FORMAT,
        "writer_identity": identity,
        "limits": limits,
        "bindings": bindings,
        "canonical_encoding": "collector-_encoded(frame)+LF",
        "compression": {
            "type": "gzip-member",
            "level": 1,
            "filename": "",
            "mtime": 0,
            "header_hex": GZIP_HEADER.hex(),
        },
        "record_prefix": "FRAM + >QII(seq,member_bytes,plaintext_bytes)",
        "trailer_prefix": "DONE + >I(canonical_trailer_bytes)",
    }


def _archive_anchors(count, head, tail, canonical_sha, canonical_bytes, identity):
    return {
        "frame_count": count,
        "first_frame_sha256": head,
        "last_frame_sha256": tail,
        "canonical_sha256": canonical_sha,
        "canonical_bytes": canonical_bytes,
        "writer_id": identity["writer_id"],
        "bindings_sha256": identity["bindings_sha256"],
        "limits_sha256": identity["limits_sha256"],
    }


def _gzip_member(line):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=buffer, mtime=0) as stream:
        stream.write(line)
    member = buffer.getvalue()
    if not member.startswith(GZIP_HEADER):
        raise ArchiveError("runtime produced an unexpected deterministic gzip header")
    return member


class ArchiveWriter:
    """One exclusive per-trial file; acknowledge only complete fsynced members."""

    def __init__(self, directory, *, limits, bindings):
        self.directory = _archive_directory(directory)
        self.bindings = _storage_bindings(bindings)
        self.limits = _storage_limits(limits, self.bindings)
        self.path = self.directory / ARCHIVE_NAME
        self._fd = None
        self._failed = None
        self._closed_metadata = None
        self._count, self._canonical_bytes, self._physical_bytes = 0, 0, 0
        self._head = self._tail = None
        self._canonical, self._physical = hashlib.sha256(), hashlib.sha256()
        try:
            self._fd = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            metadata = os.fstat(self._fd)
            identity = {
                "directory": str(self.directory),
                "archive_name": ARCHIVE_NAME,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "owner_uid": metadata.st_uid,
                "mode": metadata.st_mode,
                "bindings_sha256": _digest(self.bindings),
                "limits_sha256": _digest(self.limits),
            }
            identity["writer_id"] = _digest(identity)
            self._identity = identity
            header = _encoded(_archive_header(identity, self.limits, self.bindings))
            if len(header) > ARCHIVE_JSON_LIMIT:
                raise ArchiveError("archive header exceeds its fixed bound")
            self._header_sha = hashlib.sha256(header).hexdigest()
            longest = _archive_anchors(
                self.limits["frames"],
                "f" * 64,
                "f" * 64,
                "f" * 64,
                self.limits["canonical_archive_bytes"],
                identity,
            )
            self._footer_reserve = 8 + len(_encoded(longest))
            if (
                len(ARCHIVE_MAGIC) + 4 + len(header) + self._footer_reserve
                > self.limits["physical_archive_bytes"]
            ):
                raise ArchiveError("physical archive cap cannot fund its fixed header/trailer")
            self._write_exact(ARCHIVE_MAGIC + struct.pack(">I", len(header)) + header)
            os.fsync(self._fd)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, ValueError) as error:
            self._failed = str(error)
            self.close()
            raise ArchiveError(f"archive creation failed: {error}") from error

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def _check_handle(self):
        if self._fd is None or self._failed or self._closed_metadata is not None:
            raise ArchiveError("archive writer is failed or closed")
        if (
            _digest(self.limits) != self._identity["limits_sha256"]
            or _digest(self.bindings) != self._identity["bindings_sha256"]
        ):
            raise ArchiveError("archive sealed limits or execution bindings changed")
        opened, named = os.fstat(self._fd), os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_mode)
            != (
                self._identity["device"],
                self._identity["inode"],
                self._identity["owner_uid"],
                self._identity["mode"],
            )
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_size != self._physical_bytes
            or os.lseek(self._fd, 0, os.SEEK_CUR) != self._physical_bytes
        ):
            raise ArchiveError("bound archive file or exclusive handle changed")

    def _write_exact(self, data):
        written = os.write(self._fd, data)
        if type(written) is int and 0 <= written <= len(data):
            self._physical.update(data[:written])
            self._physical_bytes += written
        if written != len(data):
            raise ArchiveError("short archive write; no durable acknowledgement")

    def _progress(self):
        return {
            "writer_identity": self.identity,
            "frame_count": self._count,
            "first_frame_sha256": self._head,
            "last_frame_sha256": self._tail,
            "canonical_sha256": self._canonical.hexdigest(),
            "canonical_bytes": self._canonical_bytes,
            "physical_sha256": self._physical.hexdigest(),
            "physical_bytes": self._physical_bytes,
        }

    def append(self, seq, frame):
        try:
            self._check_handle()
            if type(seq) is not int or seq != self._count or seq >= self.limits["frames"]:
                raise ArchiveError("archive sequence or frame count limit changed")
            if (
                not isinstance(frame, dict)
                or frame.get("previous_frame_sha256") != self._tail
                or frame.get("frame_sha256")
                != _digest({key: value for key, value in frame.items() if key != "frame_sha256"})
            ):
                raise ArchiveError("archive frame hash or chain changed")
            line = _encoded(frame) + b"\n"
            if len(line) > self.limits["frame_plaintext_bytes"]:
                raise ArchiveError("archive plaintext frame limit exceeded")
            if self._canonical_bytes + len(line) > self.limits["canonical_archive_bytes"]:
                raise ArchiveError("canonical archive byte limit exceeded")
            member = _gzip_member(line)
            record = b"FRAM" + struct.pack(">QII", seq, len(member), len(line)) + member
            if (
                self._physical_bytes + len(record) + self._footer_reserve
                > self.limits["physical_archive_bytes"]
            ):
                raise ArchiveError("physical archive byte limit exceeded")
            self._write_exact(record)
            os.fsync(self._fd)
            self._check_handle()
            self._canonical.update(line)
            self._canonical_bytes += len(line)
            self._head = self._head or frame["frame_sha256"]
            self._tail = frame["frame_sha256"]
            self._count += 1
            return dict(self._progress(), seq=seq, frame_sha256=self._tail, durable=True)
        except (OSError, ValueError, TypeError) as error:
            self._failed = str(error)
            raise ArchiveError(f"archive append failed: {error}") from error

    def finish(self):
        if self._closed_metadata is not None:
            return copy.deepcopy(self._closed_metadata)
        try:
            self._check_handle()
            trailer = _encoded(
                _archive_anchors(
                    self._count,
                    self._head,
                    self._tail,
                    self._canonical.hexdigest(),
                    self._canonical_bytes,
                    self._identity,
                )
            )
            if (
                len(trailer) > ARCHIVE_JSON_LIMIT
                or self._physical_bytes + 8 + len(trailer) > self.limits["physical_archive_bytes"]
            ):
                raise ArchiveError("archive trailer limit exceeded")
            self._write_exact(b"DONE" + struct.pack(">I", len(trailer)) + trailer)
            os.fsync(self._fd)
            self._check_handle()
            self._closed_metadata = dict(
                self._progress(),
                format=ARCHIVE_FORMAT,
                closed=True,
                failed=False,
                bindings=copy.deepcopy(self.bindings),
                limits=copy.deepcopy(self.limits),
                header_sha256=self._header_sha,
            )
            os.close(self._fd)
            self._fd = None
            return copy.deepcopy(self._closed_metadata)
        except (OSError, ValueError, TypeError) as error:
            self._failed = str(error)
            raise ArchiveError(f"archive finish failed: {error}") from error

    def failure_metadata(self):
        return dict(
            self._progress(),
            format=ARCHIVE_FORMAT,
            closed=False,
            failed=bool(self._failed),
            error=self._failed,
            bindings=copy.deepcopy(self.bindings),
            limits=copy.deepcopy(self.limits),
            qualification=False,
        )

    def close(self):
        if self._fd is not None:
            if self._closed_metadata is None and self._failed is None:
                self._failed = "writer closed without qualified finish"
            os.close(self._fd)
            self._fd = None


class ArchiveReader:
    """Bounded single-pass reader; normal exhaustion verifies all physical and logical bytes."""

    def __init__(self, metadata, *, directory):
        self.directory = _archive_directory(directory)
        self.metadata = copy.deepcopy(metadata)
        if (
            not isinstance(metadata, dict)
            or metadata.get("closed") is not True
            or metadata.get("failed") is not False
            or metadata.get("format") != ARCHIVE_FORMAT
        ):
            raise ArchiveError("archive is not a closed qualified storage candidate")
        self.bindings = _storage_bindings(metadata.get("bindings"))
        self.limits = _storage_limits(metadata.get("limits"), self.bindings)
        self.identity = copy.deepcopy(metadata.get("writer_identity"))
        if (
            not isinstance(self.identity, dict)
            or set(self.identity)
            != {
                "directory",
                "archive_name",
                "device",
                "inode",
                "owner_uid",
                "mode",
                "bindings_sha256",
                "limits_sha256",
                "writer_id",
            }
            or any(
                type(self.identity[key]) is not int or self.identity[key] < 0
                for key in ("device", "inode", "owner_uid", "mode")
            )
            or self.identity.get("directory") != str(self.directory)
            or self.identity.get("archive_name") != ARCHIVE_NAME
            or self.identity.get("bindings_sha256") != _digest(self.bindings)
            or self.identity.get("limits_sha256") != _digest(self.limits)
            or self.identity.get("writer_id")
            != _digest({key: value for key, value in self.identity.items() if key != "writer_id"})
        ):
            raise ArchiveError("archive identity or sealed bindings changed")
        if any(
            type(metadata.get(key)) is not int or not 0 <= metadata[key] <= limit
            for key, limit in (
                ("frame_count", self.limits["frames"]),
                ("canonical_bytes", self.limits["canonical_archive_bytes"]),
                ("physical_bytes", self.limits["physical_archive_bytes"]),
            )
        ) or any(
            not isinstance(metadata.get(key), str)
            or re.fullmatch("[0-9a-f]{64}", metadata[key]) is None
            for key in ("physical_sha256", "canonical_sha256", "header_sha256")
        ):
            raise ArchiveError("archive metadata has invalid digest/count/byte anchors")
        self._consumed, self.verified = False, False

    def iter_frames(self):
        if self._consumed:
            raise ArchiveError("archive reader is single-pass and already consumed")
        self._consumed = True
        fd = None
        physical, canonical = hashlib.sha256(), hashlib.sha256()
        physical_bytes = canonical_bytes = count = 0
        head = tail = None

        def read_exact(size):
            nonlocal physical_bytes
            if size < 0 or physical_bytes + size > self.limits["physical_archive_bytes"]:
                raise ArchiveError("archive read exceeds physical limit")
            chunks, remaining = [], size
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    raise ArchiveError("truncated archive member or trailer")
                chunks.append(chunk)
                physical.update(chunk)
                physical_bytes += len(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def canonical_json(data):
            try:
                value = json.loads(data)
                if _encoded(value) != data:
                    raise ArchiveError("noncanonical archive JSON")
                return value
            except (ValueError, UnicodeError, TypeError) as error:
                raise ArchiveError("invalid canonical archive JSON") from error

        try:
            fd = os.open(self.directory / ARCHIVE_NAME, os.O_RDONLY | os.O_NOFOLLOW)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
                opened.st_uid,
                opened.st_mode,
            ) != (
                self.identity["device"],
                self.identity["inode"],
                self.identity["owner_uid"],
                self.identity["mode"],
            ):
                raise ArchiveError("archive physical file identity changed")
            if read_exact(len(ARCHIVE_MAGIC)) != ARCHIVE_MAGIC:
                raise ArchiveError("archive magic changed")
            (length,) = struct.unpack(">I", read_exact(4))
            if not 0 < length <= ARCHIVE_JSON_LIMIT:
                raise ArchiveError("archive header length exceeds limit")
            raw_header = read_exact(length)
            expected_header = _archive_header(self.identity, self.limits, self.bindings)
            if (
                canonical_json(raw_header) != expected_header
                or hashlib.sha256(raw_header).hexdigest() != self.metadata["header_sha256"]
            ):
                raise ArchiveError("archive header bindings or digest changed")
            while True:
                tag = read_exact(4)
                if tag == b"DONE":
                    (length,) = struct.unpack(">I", read_exact(4))
                    if not 0 < length <= ARCHIVE_JSON_LIMIT:
                        raise ArchiveError("archive trailer length exceeds limit")
                    trailer = canonical_json(read_exact(length))
                    if trailer != _archive_anchors(
                        count, head, tail, canonical.hexdigest(), canonical_bytes, self.identity
                    ):
                        raise ArchiveError("archive final anchors changed")
                    break
                if tag != b"FRAM":
                    raise ArchiveError("archive has an unknown record tag")
                seq, member_size, plain_size = struct.unpack(">QII", read_exact(16))
                if (
                    seq != count
                    or count >= self.limits["frames"]
                    or not 0 < plain_size <= self.limits["frame_plaintext_bytes"]
                    or not 0 < member_size <= self.limits["frame_plaintext_bytes"] + 65536
                    or canonical_bytes + plain_size > self.limits["canonical_archive_bytes"]
                ):
                    raise ArchiveError("archive member sequence or resource limit changed")
                member = read_exact(member_size)
                if not member.startswith(GZIP_HEADER):
                    raise ArchiveError("archive gzip header changed")
                decompressor = zlib.decompressobj(31)
                line = decompressor.decompress(member, plain_size + 1)
                if (
                    not decompressor.eof
                    or decompressor.unused_data
                    or decompressor.unconsumed_tail
                    or len(line) != plain_size
                    or not line.endswith(b"\n")
                ):
                    raise ArchiveError(
                        "archive member exceeds plaintext bound or has an invalid trailer"
                    )
                frame = canonical_json(line[:-1])
                if (
                    not isinstance(frame, dict)
                    or frame.get("previous_frame_sha256") != tail
                    or frame.get("frame_sha256")
                    != _digest(
                        {key: value for key, value in frame.items() if key != "frame_sha256"}
                    )
                ):
                    raise ArchiveError("archive frame hash chain changed")
                canonical.update(line)
                canonical_bytes += len(line)
                head = head or frame["frame_sha256"]
                tail = frame["frame_sha256"]
                count += 1
                yield frame
            if os.read(fd, 1):
                raise ArchiveError("archive has trailing bytes or unlisted members")
            observed = {
                "frame_count": count,
                "first_frame_sha256": head,
                "last_frame_sha256": tail,
                "canonical_sha256": canonical.hexdigest(),
                "canonical_bytes": canonical_bytes,
                "physical_sha256": physical.hexdigest(),
                "physical_bytes": physical_bytes,
            }
            if any(self.metadata.get(key) != value for key, value in observed.items()):
                raise ArchiveError(
                    "archive final physical/canonical digest, count or bytes changed"
                )
            self.verified = True
        except (OSError, ValueError, TypeError, KeyError, zlib.error) as error:
            raise ArchiveError(f"archive verification failed: {error}") from error
        finally:
            if fd is not None:
                os.close(fd)


def _identity(process):
    if (
        type(process.get("pid")) is not int
        or process["pid"] <= 0
        or type(process.get("startticks")) is not int
        or process["startticks"] < 0
    ):
        raise ValueError("invalid typed process identity")
    return f"{process['pid']}:{process['startticks']}"


def _observe_running_state(process, state):
    states = set(process.get("observed_running_states", []))
    if state in ("R", "D"):
        states.add(state)
    process["observed_running_states"] = sorted(states)
    process["active"] = bool(process.get("active", False) or states)


def _validate_configuration(policy, intent):
    def text(value):
        return isinstance(value, str) and 0 < len(value) <= 4096 and bool(value.strip())

    def sha(value):
        return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None

    def absolute(value):
        return text(value) and Path(value).is_absolute() and ".." not in Path(value).parts

    if (
        not isinstance(policy, dict)
        or not isinstance(intent, dict)
        or any(not text(policy.get(key)) for key in ("boot_id", "pid_namespace", "gpu_uuid"))
        or not isinstance(policy.get("roles"), list)
        or len(policy["roles"]) > 256
        or intent.get("kind") not in ("serving", "overhead")
        or not text(intent.get("job_id"))
        or not absolute(intent.get("output"))
        or not sha(intent.get("worker_argv_sha256"))
    ):
        raise ValueError("invalid monitor policy or execution intent")
    if (
        type(policy.get("max_evidence_bytes", 64 * 1024 * 1024)) is not int
        or not 4096 <= policy.get("max_evidence_bytes", 64 * 1024 * 1024) <= 64 * 1024 * 1024
    ):
        raise ValueError("invalid monitor retention resource policy")
    ids = set()
    for role in policy["roles"]:
        if (
            not isinstance(role, dict)
            or not text(role.get("id"))
            or role["id"] in ids
            or role["id"] == "unknown"
            or role["id"].startswith("owned-")
            or role.get("activity") not in ("idle", "permitted")
            or ("kernel" in role and type(role["kernel"]) is not bool)
            or (
                role.get("kernel") is not True
                and role.get("identity_basis", "direct") == "direct"
                and (
                    not absolute(role.get("exe"))
                    or not sha(role.get("argv_sha256"))
                    or not absolute(role.get("cgroup"))
                )
            )
        ):
            raise ValueError("invalid or ambiguous monitor role policy")
        basis = role.get("identity_basis", "direct")
        for option in ("early_boot", "main_subgroup"):
            if option in role and (
                type(role[option]) is not bool
                or policy.get("context_decision_sha256") != OS_CONTEXT_DECISION
            ):
                raise ValueError("OS compatibility option requires the adopted context decision")
        if role.get("early_boot") and not (
            basis == "boot-manager" or (role.get("kernel") is True and role.get("identities"))
        ):
            raise ValueError("early-boot permission needs an exact boot-manager or kernel identity")
        if role.get("main_subgroup") and basis != "systemd-main":
            raise ValueError("only a fixed systemd MainPID may use its exact declared subgroup")
        if basis not in (
            "direct",
            "systemd-main",
            "boot-manager",
            "os-fixed-helper",
            "baseline-terminal-z",
            "os-telemetry-helper",
            "os-containerd-shim",
        ):
            raise ValueError("unknown OS role identity basis")
        if (
            basis
            in (
                "os-fixed-helper",
                "baseline-terminal-z",
                "os-telemetry-helper",
                "os-containerd-shim",
            )
            and policy.get("context_decision_sha256") != OS_CONTEXT_DECISION
        ):
            raise ValueError("OS context requires the adopted lifecycle decision")
        if basis != "direct":
            proof = role.get("corroboration")
            if (
                policy.get("decision_sha256") != OS_IDENTITY_DECISION
                or role.get("kernel") is True
                or role["activity"] != "permitted"
                or not text(role.get("exe_unavailable_reason"))
                or not isinstance(proof, dict)
                or proof.get("boot_id") != policy["boot_id"]
                or proof.get("pid_namespace") != policy["pid_namespace"]
                or (
                    basis not in ("os-telemetry-helper", "os-containerd-shim")
                    and any(
                        type(proof.get(key)) is not int or proof[key] <= 0
                        for key in ("pid", "pgid", "sid")
                    )
                )
                or (
                    basis not in ("os-telemetry-helper", "os-containerd-shim")
                    and (
                        type(proof.get("startticks")) is not int
                        or proof["startticks"] < (0 if role.get("early_boot") else 1)
                    )
                )
                or not isinstance(proof.get("uids"), list)
                or len(proof["uids"]) != 4
                or any(type(uid) is not int or uid < 0 for uid in proof["uids"])
                or not text(proof.get("comm"))
                or (
                    basis not in ("os-telemetry-helper", "os-containerd-shim")
                    and not sha(proof.get("argv_sha256"))
                    and not (
                        basis == "baseline-terminal-z"
                        and proof.get("argv_sha256") is None
                        and text(proof.get("argv_unavailable_reason"))
                    )
                )
                or not absolute(proof.get("cgroup"))
            ):
                raise ValueError("invalid approved OS role corroboration policy")
            if basis != "baseline-terminal-z":
                _validate_file_proof(proof.get("configured_executable"))
                if "configured_alias" in proof:
                    alias = proof["configured_alias"]
                    if (
                        policy.get("context_decision_sha256") != OS_CONTEXT_DECISION
                        or not isinstance(alias, dict)
                        or not absolute(alias.get("path"))
                        or alias.get("target") != proof["configured_executable"]["path"]
                        or not isinstance(alias.get("components"), list)
                        or not 1 <= len(alias["components"]) <= 128
                    ):
                        raise ValueError(
                            "configured alias requires its adopted complete protected proof"
                        )
                    for entry in alias["components"]:
                        if (
                            not isinstance(entry, dict)
                            or not absolute(entry.get("path"))
                            or not isinstance(entry.get("stat"), dict)
                            or set(entry["stat"]) != {"device", "inode", "uid", "mode"}
                            or any(
                                type(value) is not int or value < 0
                                for value in entry["stat"].values()
                            )
                            or entry["stat"]["uid"] != 0
                        ):
                            raise ValueError(
                                "configured alias has malformed protected component metadata"
                            )
                inputs = proof.get("configured_inputs", [])
                if not isinstance(inputs, list) or len(inputs) > 8:
                    raise ValueError("OS configured inputs must be a finite protected list")
                for configured in inputs:
                    _validate_file_proof(configured)
                name = Path(proof["configured_executable"]["path"]).name
                if (
                    name.startswith("python") or name in ("sh", "bash", "dash", "node")
                ) and not inputs:
                    raise ValueError(
                        "interpreted OS recipe requires its protected script/input proof"
                    )
            elif type(proof.get("cpu_ticks")) is not int or proof["cpu_ticks"] < 0:
                raise ValueError("terminal baseline needs its fixed measured CPU counters")
            if basis == "boot-manager":
                if (
                    proof["pid"] != 1
                    or proof["uids"] != [0, 0, 0, 0]
                    or proof.get("manager") != "systemd"
                    or proof["comm"] != "systemd"
                    or Path(proof["configured_executable"]["path"]).name != "systemd"
                ):
                    raise ValueError("boot-manager role must identify the exact namespace PID1")
            elif basis == "systemd-main":
                unit = proof.get("unit")
                if (
                    not isinstance(unit, dict)
                    or not isinstance(unit.get("name"), str)
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service", unit["name"])
                    is None
                    or not isinstance(unit.get("metadata"), dict)
                    or not sha(unit.get("metadata_sha256"))
                ):
                    raise ValueError("systemd-main role requires pinned typed unit metadata")
                _validate_file_proof(unit.get("file"), unit=True)
            elif basis == "os-fixed-helper":
                parent = role.get("parent_identity")
                if (
                    not text(role.get("parent_role"))
                    or not isinstance(parent, dict)
                    or any(
                        type(parent.get(key)) is not int
                        or parent[key] < (0 if key == "startticks" else 1)
                        for key in ("pid", "startticks", "pgid", "sid")
                    )
                    or not isinstance(parent.get("uids"), list)
                    or len(parent["uids"]) != 4
                    or any(type(uid) is not int or uid < 0 for uid in parent["uids"])
                    or not text(parent.get("comm"))
                    or not sha(parent.get("argv_sha256"))
                    or not absolute(parent.get("cgroup"))
                ):
                    raise ValueError("fixed OS child needs its full approved parent identity")
            elif basis == "os-telemetry-helper":
                argv = proof.get("argv")
                if (
                    not text(role.get("parent_role"))
                    or role.get("root_unit") != "dgx-dashboard-admin.service"
                    or role.get("session") not in ("new", "inherit")
                    or type(role.get("max_live")) is not int
                    or not 1 <= role["max_live"] <= 16
                    or not isinstance(argv, list)
                    or not 1 <= len(argv) <= 128
                    or any(
                        not (
                            (isinstance(token, str) and "\0" not in token)
                            or token == {"source": "parent.pid"}
                        )
                        for token in argv
                    )
                ):
                    raise ValueError("telemetry requires a finite materialized exact DGX recipe")
            elif basis == "os-containerd-shim" and (
                intent["kind"] != "serving"
                or not text(role.get("daemon_role"))
                or role.get("namespace") != "moby"
                or not absolute(role.get("address"))
                or proof.get("argv")
                != [
                    proof["configured_executable"]["path"],
                    "-namespace",
                    role["namespace"],
                    "-id",
                    {"source": "registered.cid"},
                    "-address",
                    role["address"],
                ]
            ):
                raise ValueError(
                    "shim requires its exact registered Docker namespace/address template"
                )
        if "allow_new_kernel" in role and (
            type(role["allow_new_kernel"]) is not bool
            or (
                role["allow_new_kernel"]
                and (role.get("kernel") is not True or role["activity"] != "permitted")
            )
        ):
            raise ValueError("new kernel role needs an explicit verified kernel policy")
        if "identities" in role and (
            not isinstance(role["identities"], list)
            or any(
                not isinstance(identity, dict)
                or any(
                    type(identity.get(key)) is not int
                    or identity[key] < (0 if key == "startticks" and role.get("early_boot") else 1)
                    for key in ("pid", "startticks")
                )
                for identity in role["identities"]
            )
        ):
            raise ValueError("role policy has invalid typed process identities")
        if "io_unavailable_reason" in role and not text(role["io_unavailable_reason"]):
            raise ValueError("role policy needs explicit I/O visibility provenance")
        ids.add(role["id"])
    if any(
        role.get("parent_role") not in ids or role.get("parent_role") == role["id"]
        for role in policy["roles"]
        if role.get("identity_basis") in ("os-fixed-helper", "os-telemetry-helper")
    ):
        raise ValueError("fixed OS child has an unknown parent role")
    if any(
        role.get("daemon_role") not in ids
        for role in policy["roles"]
        if role.get("identity_basis") == "os-containerd-shim"
    ):
        raise ValueError("registered shim has no approved daemon role")
    if "io_unavailable_reason" in intent and not text(intent["io_unavailable_reason"]):
        raise ValueError("execution intent needs explicit I/O visibility provenance")
    helpers = intent.get("helpers", [])
    if not isinstance(helpers, list) or len(helpers) > 128:
        raise ValueError("invalid helper intent list")
    parents, signatures = {}, set()
    for helper in helpers:
        if (
            not isinstance(helper, dict)
            or not text(helper.get("id"))
            or helper["id"] == "worker"
            or helper["id"] in parents
            or not text(helper.get("parent"))
            or not absolute(helper.get("exe"))
            or not sha(helper.get("argv_sha256"))
            or helper.get("session", "inherit") not in ("new", "inherit")
            or (
                "uids" in helper
                and (
                    not isinstance(helper["uids"], list)
                    or len(helper["uids"]) != 4
                    or any(type(uid) is not int or uid < 0 for uid in helper["uids"])
                )
            )
            or ("cgroup" in helper and not absolute(helper["cgroup"]))
        ):
            raise ValueError("invalid helper identity or session intent")
        signature = tuple(
            helper.get(key) for key in ("parent", "exe", "argv_sha256", "session", "cgroup")
        )
        if signature in signatures:
            raise ValueError("ambiguous duplicate helper recipe")
        signatures.add(signature)
        parents[helper["id"]] = helper["parent"]
    for helper in parents:
        visited, current = set(), helper
        while current != "worker":
            if current in visited or current not in parents:
                raise ValueError("helper intent has an unknown parent or cycle")
            visited.add(current)
            current = parents[current]
    if intent["kind"] == "serving":
        server = intent.get("server")
        if (
            not isinstance(server, dict)
            or not text(server.get("name"))
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", server["name"]) is None
            or not text(server.get("image"))
            or not isinstance(server.get("labels"), dict)
            or server["labels"].get("xgraph.serving-output") != str(Path(intent["output"]).parent)
            or any(not text(key) or not text(value) for key, value in server["labels"].items())
            or (
                "image_id" in server
                and (
                    not isinstance(server["image_id"], str)
                    or re.fullmatch("sha256:[0-9a-f]{64}", server["image_id"]) is None
                )
            )
        ):
            raise ValueError("invalid serving container source intent")


def _validate_file_proof(proof, *, unit=False):
    if (
        not isinstance(proof, dict)
        or not isinstance(proof.get("path"), str)
        or not Path(proof["path"]).is_absolute()
        or ".." in Path(proof["path"]).parts
        or not isinstance(proof.get("stat"), dict)
        or not isinstance(proof.get("ancestors"), list)
    ):
        raise ValueError("OS role needs canonical protected file provenance")
    expected = proof["stat"]
    keys = {"device", "inode", "uid", "mode", "size", "mtime_ns"}
    if (
        set(expected) != keys
        or any(type(value) is not int or value < 0 for value in expected.values())
        or expected["uid"] != 0
        or expected["mode"] & 0o022
        or not stat.S_ISREG(expected["mode"])
    ):
        raise ValueError("OS role file is not pinned as root-owned and protected")
    parents = [str(parent) for parent in reversed(Path(proof["path"]).parents)]
    if (
        any(not isinstance(entry, dict) for entry in proof["ancestors"])
        or [entry.get("path") for entry in proof["ancestors"]] != parents
    ):
        raise ValueError("OS role file needs its complete canonical ancestor chain")
    for entry in proof["ancestors"]:
        pin = entry.get("stat")
        if (
            not isinstance(pin, dict)
            or set(pin) != {"device", "inode", "uid", "mode"}
            or any(type(value) is not int or value < 0 for value in pin.values())
            or pin["uid"] != 0
            or pin["mode"] & 0o022
            or not stat.S_ISDIR(pin["mode"])
        ):
            raise ValueError("OS role file has an untrusted ancestor policy")
    if unit and (
        expected["size"] > 65536
        or not isinstance(proof.get("sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", proof["sha256"]) is None
    ):
        raise ValueError("OS unit requires a bounded source hash")


def _iter_reconstructed_evidence(evidence):
    if evidence.get("encoding") != "inventory-delta-v1":
        raise ValueError("unknown inventory encoding")
    baseline = evidence["baseline"]
    if "policy" in baseline:
        policy_sha = hashlib.sha256(
            json.dumps(baseline["policy"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if baseline.get("policy_sha256") != policy_sha:
            raise ValueError("frozen baseline policy digest changed")
    if "intent" in baseline and baseline.get("intent_sha256") != _digest(baseline["intent"]):
        raise ValueError("frozen baseline intent digest changed")
    if "archive" in evidence:
        archive = evidence["archive"]
        if (
            not isinstance(archive, dict)
            or archive.get("closed") is not True
            or archive.get("failed") is not False
            or archive.get("format") != ARCHIVE_FORMAT
            or archive.get("bindings")
            != _policy_storage_bindings(baseline.get("policy", {}), baseline.get("intent", {}))
            or archive.get("limits")
            != _storage_limits(
                baseline.get("policy", {}).get("storage_limits"), archive.get("bindings")
            )
            or archive.get("frame_count") != evidence.get("frame_count")
            or archive.get("last_frame_sha256") != evidence.get("last_frame_sha256")
        ):
            raise ArchiveError(
                "archive evidence bindings/limits/final anchors differ from frozen baseline"
            )
    indices = baseline["sample_indices"]
    if (
        not isinstance(indices, list)
        or len(indices) != 2
        or any(type(index) is not int or index < 0 for index in indices)
        or indices[1] != indices[0] + 1
        or len(baseline["snapshots"]) != 2
    ):
        raise ValueError("invalid baseline snapshot indices")
    expected_baseline = dict(zip(indices, baseline["snapshots"], strict=True))
    previous, registry, seen_baseline = {}, {}, set()
    inventory_sha, frame_sha = _digest([]), None
    frame_count, last_at = 0, None
    for index, frame in enumerate(evidence["samples"]):
        unsigned = {key: value for key, value in frame.items() if key != "frame_sha256"}
        if (
            frame["frame_sha256"] != _digest(unsigned)
            or frame["previous_frame_sha256"] != frame_sha
            or frame["previous_inventory_sha256"] != inventory_sha
        ):
            raise ValueError("inventory frame digest chain changed")
        rows = {_identity(row): row for row in frame["processes"]}
        refs, identities = frame["references"], frame["identities"]
        if (
            len(rows) != len(frame["processes"])
            or len(set(refs)) != len(refs)
            or len(set(identities)) != len(identities)
            or set(rows) & set(refs)
            or set(rows) | set(refs) != set(identities)
            or type(frame["process_count"]) is not int
            or frame["process_count"] != len(identities)
            or frame["created"] != sorted(set(identities) - previous.keys())
            or frame["exited"] != sorted(previous.keys() - set(identities))
        ):
            raise ValueError("inventory identity or lifecycle changed")
        for key in refs:
            row = previous.get(key)
            if (
                row is None
                or row["active"] is not False
                or row["gpu"] is not False
                or row["classification"] == "unknown"
                or row["classification"].startswith("owned-")
                or frame["complete"] is not True
                or frame["errors"]
                or frame["races"]
            ):
                raise ValueError("invalid inventory identity reference")
            rows[key] = row
        processes = [rows[key] for key in identities]
        if frame["inventory_sha256"] != _digest(processes):
            raise ValueError("reconstructed inventory digest changed")
        snapshot = {
            key: value
            for key, value in frame.items()
            if key
            not in (
                "frame_sha256",
                "previous_frame_sha256",
                "previous_inventory_sha256",
                "inventory_sha256",
                "references",
                "identities",
                "process_count",
                "created",
                "exited",
            )
        }
        snapshot["processes"] = processes
        for key in frame["created"]:
            registry[key] = {
                "pid": rows[key]["pid"],
                "startticks": rows[key]["startticks"],
                "boot_id": snapshot["boot_id"],
                "pid_namespace": snapshot["pid_namespace"],
            }
        if index in expected_baseline:
            if _digest(snapshot) != _digest(expected_baseline[index]):
                raise ValueError("baseline snapshot digest changed")
            seen_baseline.add(index)
        previous, inventory_sha, frame_sha = rows, frame["inventory_sha256"], frame["frame_sha256"]
        frame_count, last_at = index + 1, snapshot["at"]
        yield snapshot
    if _digest(registry) != _digest(evidence["identity_registry"]):
        raise ValueError("inventory identity registry changed")
    if seen_baseline != set(indices):
        raise ValueError("baseline snapshot sequence is incomplete")
    if "frame_count" in evidence and (
        type(evidence["frame_count"]) is not int or evidence["frame_count"] != frame_count
    ):
        raise ValueError("inventory frame count is incomplete")
    if "last_frame_sha256" in evidence and evidence["last_frame_sha256"] != frame_sha:
        raise ValueError("last inventory frame digest changed")
    if (
        type(last_at) not in (int, float)
        or not math.isfinite(last_at)
        or last_at < evidence["interval"]["end"]
    ):
        raise ValueError("inventory interval has an incomplete final boundary")


def iter_reconstructed_evidence(evidence):
    """Yield independent snapshots; only successful exhaustion verifies the full archive."""
    for snapshot in _iter_reconstructed_evidence(evidence):
        yield copy.deepcopy(snapshot)


def reconstruct_evidence(evidence):
    """Compatibility convenience for small archives; use the iterator for long intervals."""
    return copy.deepcopy(list(_iter_reconstructed_evidence(evidence)))


def readonly_command_runner(argv):
    docker = (
        isinstance(argv, tuple)
        and len(argv) == 7
        and argv[:6] == ("docker", "inspect", "--type", "container", "--format", DOCKER_FORMAT)
        and isinstance(argv[-1], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", argv[-1]) is not None
    )
    systemd = (
        isinstance(argv, tuple)
        and len(argv) == 6
        and argv[:5] == SYSTEMD_PREFIX
        and isinstance(argv[-1], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service", argv[-1]) is not None
    )
    if argv not in (GPU_QUERY, COMPUTE_QUERY) and not docker and not systemd:
        raise ValueError("only the fixed read-only inventory queries are permitted")
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("DBUS_", "SYSTEMD_"))
    }
    environment.update(LC_ALL="C", SYSTEMD_PAGER="", SYSTEMD_COLORS="0", SYSTEMD_URLIFY="0")
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            env=environment,
        )
    except (OSError, TimeoutExpired) as error:
        raise ValueError(f"read-only inventory query unavailable: {argv[0]}") from error
    if result.returncode:
        missing = (
            docker
            and result.returncode == 1
            and not result.stdout.strip()
            and re.fullmatch(
                r"(?:Error:|Error response from daemon:)\s*No such (?:object|container):\s*"
                + re.escape(argv[-1])
                + r"\s*",
                result.stderr,
            )
            is not None
        )
        if missing:
            return "null"
        raise ValueError(f"read-only inventory query failed: {argv[0]} exit {result.returncode}")
    if not isinstance(result.stdout, str) or len(result.stdout.encode()) > 1024 * 1024:
        raise ValueError("oversized or invalid read-only inventory response")
    return result.stdout


class Monitor:
    def __init__(
        self,
        policy,
        intent,
        *,
        proc_root=Path("/proc"),
        clock=time.monotonic,
        query=readonly_command_runner,
        file_stat=os.stat,
        frame_sink=None,
    ):
        _validate_configuration(policy, intent)
        if frame_sink is None and "storage_limits" in policy:
            raise ArchiveError("archive policy requires its bound writer; no fixture fallback")
        self._policy, self._intent = copy.deepcopy(policy), copy.deepcopy(intent)
        self.proc_root, self.clock, self.query = Path(proc_root), clock, query
        self._file_stat = file_stat
        self._proof_cache = {}
        self._previous = {}
        self._samples = deque(maxlen=2)
        self._frames = []
        self._frame_sizes = []
        self._identity_registry = {}
        self._inventory = {}
        self._inventory_sha = _digest([])
        self._frame_sha = None
        self._query_records = []
        self._retained_bytes = 0
        self._retention_failed = False
        self._errors = []
        self._baseline = None
        self._permitted = {}
        self._worker = None
        self._owned = {}
        self._container = None
        self._violations = []
        self._frame_count = 0
        self._first_frame_sha = None
        self._canonical_stream = hashlib.sha256()
        self._canonical_bytes = 0
        self._frame_sink = frame_sink
        self._archive_failed = None
        self._archive_metadata = None
        self._archive_physical_bytes = 0
        if frame_sink is not None:
            self._archive_bindings = _policy_storage_bindings(self._policy, self._intent)
            self._archive_limits = _storage_limits(
                self._policy.get("storage_limits"), self._archive_bindings
            )
            if (
                not isinstance(frame_sink, ArchiveWriter)
                or frame_sink.limits != self._archive_limits
                or frame_sink.bindings != self._archive_bindings
            ):
                raise ArchiveError(
                    "archive writer does not match the explicit policy/intent bindings and limits"
                )
            self._archive_identity = frame_sink.identity
            if self._archive_identity["limits_sha256"] != _digest(
                self._archive_limits
            ) or self._archive_identity["bindings_sha256"] != _digest(self._archive_bindings):
                raise ArchiveError(
                    "archive writer identity does not bind policy limits and execution"
                )

    def _state_proxy_bytes(self):
        """Canonical current-state proxy, deliberately distinct from heap/RSS or archive bytes."""
        return len(
            _encoded(
                {
                    "policy": self._policy,
                    "intent": self._intent,
                    "samples": list(self._samples),
                    "pending_frames": self._frames,
                    "inventory": self._inventory,
                    "previous": list(self._previous.values()),
                    "registry": self._identity_registry,
                    "permitted": list(self._permitted.values()),
                    "owned": list(self._owned.values()),
                    "baseline": self._baseline,
                    "worker": self._worker,
                    "container": self._container,
                    "queries": self._query_records,
                    "proof_cache": list(self._proof_cache.values()),
                    "errors": self._errors,
                    "violations": self._violations,
                    "frame_sizes": self._frame_sizes,
                    "frame_count": self._frame_count,
                    "last_frame_sha256": self._frame_sha,
                    "canonical_sha256": self._canonical_stream.hexdigest(),
                    "canonical_bytes": self._canonical_bytes,
                }
            )
        )

    def _check_state_budget(self):
        self._retained_bytes = self._state_proxy_bytes()
        if self._retained_bytes > self._archive_limits["state_proxy_bytes"]:
            raise ArchiveError("current-state canonical proxy resource budget exceeded")

    def _check_archive_progress(self, progress):
        expected = {
            "writer_identity": self._archive_identity,
            "frame_count": self._frame_count,
            "first_frame_sha256": self._first_frame_sha,
            "last_frame_sha256": self._frame_sha,
            "canonical_sha256": self._canonical_stream.hexdigest(),
            "canonical_bytes": self._canonical_bytes,
        }
        if (
            self._frame_sink.identity != self._archive_identity
            or self._frame_sink.limits != self._archive_limits
            or self._frame_sink.bindings != self._archive_bindings
            or not isinstance(progress, dict)
            or any(progress.get(key) != value for key, value in expected.items())
            or any(type(progress.get(key)) is not int for key in ("frame_count", "canonical_bytes"))
            or type(progress.get("physical_bytes")) is not int
            or not self._archive_physical_bytes
            < progress["physical_bytes"]
            <= self._archive_limits["physical_archive_bytes"]
            or not isinstance(progress.get("physical_sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", progress["physical_sha256"]) is None
        ):
            raise ArchiveError("archive acknowledgement/final anchors or bound writer changed")

    def storage_failure_metadata(self):
        if self._frame_sink is None:
            raise ArchiveError("monitor has no archive writer")
        return {
            "qualification": False,
            "error": self._archive_failed,
            "pending_frames": copy.deepcopy(self._frames),
            "writer": self._frame_sink.failure_metadata(),
            "state_proxy_bytes": self._retained_bytes,
            "last_observed_snapshot": copy.deepcopy(self._samples[-1]) if self._samples else None,
        }

    def finish_archive(self):
        if self._frame_sink is None or self._archive_failed or self._retention_failed:
            raise ArchiveError("archive is absent or failed; qualified finish is unavailable")
        if self._archive_metadata is not None:
            return copy.deepcopy(self._archive_metadata)
        try:
            if self._frames:
                raise ArchiveError("unacknowledged archive frames prevent finish")
            metadata = self._frame_sink.finish()
            self._check_archive_progress(metadata)
            if (
                metadata.get("closed") is not True
                or metadata.get("failed") is not False
                or metadata.get("format") != ARCHIVE_FORMAT
                or metadata.get("limits") != self._archive_limits
                or metadata.get("bindings") != self._archive_bindings
                or metadata.get("header_sha256")
                != _digest(
                    _archive_header(
                        self._archive_identity, self._archive_limits, self._archive_bindings
                    )
                )
            ):
                raise ArchiveError("archive finish bindings, limits or header changed")
            self._archive_physical_bytes = metadata["physical_bytes"]
            self._archive_metadata = copy.deepcopy(metadata)
            return copy.deepcopy(metadata)
        except (OSError, ValueError, TypeError) as error:
            self._archive_failed, self._retention_failed = str(error), True
            raise ArchiveError(f"archive finish failed: {error}") from error

    def bind_worker(self, worker):
        if self._baseline is None or self._worker is not None:
            raise ValueError("worker requires one frozen baseline and cannot be rebound")
        if (
            worker.get("job_id") != self._intent["job_id"]
            or worker.get("argv_sha256") != self._intent["worker_argv_sha256"]
            or any(
                type(worker.get(key)) is not int or worker[key] <= 0
                for key in ("pid", "startticks", "pgid", "sid")
            )
        ):
            raise ValueError("worker intent or process identity changed")
        process = self._process(worker["pid"])
        if process["exe"] is None:
            raise ValueError("worker executable identity must be directly readable")
        if (
            any(
                process[key] != worker[key]
                for key in ("pid", "startticks", "pgid", "sid", "argv_sha256")
            )
            or process["pgid"] != process["pid"]
            or process["sid"] != process["pid"]
        ):
            raise ValueError("worker is not the declared independent process group")
        self._qualify_io(process, self._intent.get("io_unavailable_reason"))
        self._worker = {
            key: process[key]
            for key in (
                "pid",
                "startticks",
                "pgid",
                "sid",
                "uids",
                "comm",
                "argv_sha256",
                "exe",
                "cgroup",
            )
        }
        self._worker.update(
            bound=True,
            boot_id=self._policy["boot_id"],
            job_id=self._intent["job_id"],
            origin="worker",
        )
        self._owned[(process["pid"], process["startticks"])] = copy.deepcopy(self._worker)
        return copy.deepcopy(self._worker)

    def confirm_cleanup(self):
        if self._worker is None:
            return {"confirmed_absent": False, "errors": ["worker was never bound"], "owned": []}
        snapshot = self.sample()
        errors = []
        if not snapshot["ownership_complete"]:
            errors.append("owned-resource absence cannot be established from this inventory")
        current = {
            (process["pid"], process["startticks"]): process
            for process in snapshot["processes"]
            if process.get("present_at_end", True)
        }
        if current.keys() & self._owned.keys():
            errors.append("owned process identity remains present")
        groups = {
            (owned["pid"], owned["startticks"], owned["pgid"], owned["sid"])
            for owned in self._owned.values()
            if owned["pid"] == owned["pgid"] == owned["sid"]
        }
        for pid, startticks, pgid, sid in groups:
            root = next((process for process in current.values() if process["pid"] == pid), None)
            reused = root is not None and root["startticks"] != startticks
            if not reused and any(
                process["pgid"] == pgid and process["sid"] == sid for process in current.values()
            ):
                errors.append(f"owned process group remains present: {pgid}")
        if self._intent["kind"] == "serving":
            if snapshot.get("container_state") not in ("absent", "exited"):
                errors.append("owned container absence is unconfirmed")
            if self._container is not None and any(
                process["cgroup"] == self._container["cgroup"]
                or process["cgroup"].startswith(self._container["cgroup"] + "/")
                for process in current.values()
            ):
                errors.append("owned container cgroup remains present")
        return {
            "confirmed_absent": not errors,
            "at": snapshot["at"],
            "errors": errors,
            "violations": copy.deepcopy(snapshot["errors"]),
            "owned": copy.deepcopy(list(self._owned.values())),
            "container_binding": copy.deepcopy(self._container),
        }

    def evidence(self, start, end):
        if self._frame_sink is not None and (
            self._archive_failed or self._archive_metadata is None
        ):
            raise ArchiveError(
                "archive must finish successfully before supplying qualified evidence"
            )
        if (
            self._baseline is None
            or type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not self._baseline["at"] <= start <= end
            or not self._samples
            or type(self._samples[-1]["at"]) not in (int, float)
            or end > self._samples[-1]["at"]
        ):
            raise ValueError("evidence needs frozen baseline and observed interval boundaries")
        header = {
            "frame_count": self._frame_count,
            "last_frame_sha256": self._frame_sha,
            "encoding": "inventory-delta-v1",
            "identity_registry": self._identity_registry,
            "retained_json_bytes": self._retained_bytes,
            "retention_budget_bytes": self.retention_limit_bytes,
            "permitted": list(self._permitted.values()),
            "owned": list(self._owned.values()),
            "baseline": self._baseline,
            "container_binding": self._container,
            "errors": self._errors,
            "violations": self._violations,
            "interval": {"start": start, "end": end},
            "scope": "sampled coordinated inventories; no proof of zero interference",
        }
        if self._frame_sink is None:
            header["samples"] = self._frames
        else:
            header["archive"] = self._archive_metadata
            header["retention_scope"] = (
                "current-state canonical proxy; archived wire bytes are separate"
            )
        return copy.deepcopy(header)

    @property
    def retention_limit_bytes(self):
        if self._frame_sink is not None:
            return self._archive_limits["state_proxy_bytes"]
        return self._policy.get("max_evidence_bytes", 64 * 1024 * 1024)

    def retention_estimate(self, duration_s, *, interval_s=1):
        """Project retained metadata with an allowance, without an RSS or admission claim."""
        if self._baseline is None or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (duration_s, interval_s)
        ):
            raise ValueError("retention projection needs a baseline and positive finite intervals")
        boundary = self._baseline["sample_indices"][-1] + 1
        postbaseline = self._frame_sizes[boundary:]
        observed_max = max(postbaseline, default=max(self._frame_sizes, default=0))
        allowance = 2 * observed_max + 1024
        samples = math.ceil(duration_s / interval_s) + 1
        projected = self._retained_bytes + samples * allowance
        identities = list(self._inventory)
        repeated_membership_heap = (
            2 * sys.getsizeof([])
            + sum(sys.getsizeof(key) for key in identities)
            + 2 * len(identities) * struct.calcsize("P")
        )
        result = {
            "retention_budget_bytes": self.retention_limit_bytes,
            "retained_json_bytes": self._retained_bytes,
            "baseline_process_count": len(self._baseline["snapshots"][-1]["processes"]),
            "duration_seconds": duration_s,
            "nominal_interval_seconds": interval_s,
            "interval_samples_including_boundaries": samples,
            "observed_postbaseline_frames": len(postbaseline),
            "observed_max_frame_json_bytes": observed_max,
            "baseline_full_frame_max_bytes": max(
                self._frame_sizes[index] for index in self._baseline["sample_indices"]
            ),
            "frame_allowance_bytes": allowance,
            "allowance_basis": (
                "2 * observed maximum frame JSON bytes + 1024; engineering allowance, not a bound"
            ),
            "projected_json_bytes": projected,
            "projection_is_bound": False,
            "repeated_membership_heap_component_bytes_per_frame": repeated_membership_heap,
            "projected_repeated_membership_heap_component_bytes": samples
            * repeated_membership_heap,
            "heap_component_scope": (
                "One current identity string per PID plus two list headers/reference vectors; "
                "excludes rows, proofs, queries, frame dictionaries, allocator and future "
                "process growth; not a total heap/RSS bound."
            ),
            "capacity_status": (
                "PROJECTED_FIT_NOT_ADMISSION"
                if postbaseline and projected <= self.retention_limit_bytes
                else "NOT_QUALIFIED"
            ),
            "rss_prediction_bytes": None,
            "limitations": (
                "Future process growth/activity/metadata can exceed this assumed shape; "
                "Python RSS is not serialized bytes; no cadence thinning or probability claim."
            ),
        }
        if self._frame_sink is not None:
            result.update(
                storage_mode=ARCHIVE_FORMAT,
                storage_limits=copy.deepcopy(self._archive_limits),
                capacity_status="NOT_QUALIFIED",
                projected_json_bytes=None,
                projected_canonical_archive_bytes=self._canonical_bytes + samples * allowance,
                projected_physical_archive_bytes=None,
                archive_capacity_reason=(
                    "Complete qualified shape/compression/fsync and four-cell disk funding "
                    "are required before admission"
                ),
            )
        return result

    def freeze_baseline(self, samples):
        if self._baseline is not None:
            raise ValueError("baseline is already frozen; no permission reset")
        if (
            len(samples) != 2
            or len(self._samples) < 2
            or _digest(samples) != _digest(list(self._samples))
            or any(
                not sample["complete"] or sample["errors"] or sample["races"] for sample in samples
            )
            or not 0 < samples[1]["at"] - samples[0]["at"] <= 5
        ):
            raise ValueError("baseline requires two actual complete boundary snapshots")
        first = {(row["pid"], row["startticks"]) for row in samples[0]["processes"]}
        latest = {(row["pid"], row["startticks"]) for row in samples[1]["processes"]}
        if first - latest:
            raise ValueError("first-only baseline identity has no frozen permission")
        for process in samples[-1]["processes"]:
            role = self._role(process)
            if role is None or (role["activity"] == "idle" and process["active"]):
                raise ValueError("baseline contains an unqualified idle or unknown role")
            if role.get("identity_basis") == "baseline-terminal-z":
                prior = next(
                    (
                        row
                        for row in samples[0]["processes"]
                        if (row["pid"], row["startticks"])
                        == (process["pid"], process["startticks"])
                    ),
                    None,
                )
                if (
                    prior is None
                    or prior["state"] != "Z"
                    or process["state"] != "Z"
                    or prior["cpu_ticks"] != process["cpu_ticks"]
                    or prior["gpu"]
                    or process["gpu"]
                ):
                    raise ValueError(
                        "terminal permission requires two matching actual terminal baselines"
                    )
            key = process["pid"], process["startticks"]
            self._permitted[key] = self._permission_descriptor(process, role)
        self._baseline = {
            "frozen": True,
            "at": samples[-1]["at"],
            "boot_id": self._policy["boot_id"],
            "pid_namespace": self._policy["pid_namespace"],
            "gpu_uuid": self._policy["gpu_uuid"],
            "policy": copy.deepcopy(self._policy),
            "intent": copy.deepcopy(self._intent),
            "intent_sha256": _digest(self._intent),
            "snapshots": copy.deepcopy(samples),
            "sample_indices": [self._frame_count - 2, self._frame_count - 1],
            "policy_sha256": hashlib.sha256(
                json.dumps(self._policy, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "permitted": list(self._permitted.values()),
        }
        if self._frame_sink is not None:
            try:
                self._check_state_budget()
            except ArchiveError as error:
                self._archive_failed, self._retention_failed = str(error), True
                raise
        else:
            self._retained_bytes += len(_encoded(self._baseline))
        if self._retained_bytes > self.retention_limit_bytes:
            self._retention_failed = True
            raise ValueError("baseline exceeds the lossless retention resource budget")
        return copy.deepcopy(self._baseline)

    def _retain(self, record):
        current = {_identity(process): process for process in record["processes"]}
        rows, references = [], []
        for key, process in current.items():
            permitted = (process["pid"], process["startticks"]) in self._permitted
            if (
                record["complete"]
                and permitted
                and process == self._inventory.get(key)
                and process["active"] is False
                and process["gpu"] is False
                and not process["classification"].startswith("owned-")
            ):
                references.append(key)
            else:
                rows.append(copy.deepcopy(process))
        created = sorted(current.keys() - self._inventory.keys())
        for key in created:
            self._identity_registry[key] = {
                "pid": current[key]["pid"],
                "startticks": current[key]["startticks"],
                "boot_id": record["boot_id"],
                "pid_namespace": record["pid_namespace"],
            }
            if self._frame_sink is None:
                self._retained_bytes += len(_encoded(self._identity_registry[key]))
        frame = {key: copy.deepcopy(value) for key, value in record.items() if key != "processes"}
        frame.update(
            processes=rows,
            references=references,
            identities=list(current),
            created=created,
            exited=sorted(self._inventory.keys() - current.keys()),
            process_count=len(current),
            inventory_sha256=_digest(record["processes"]),
            previous_inventory_sha256=self._inventory_sha,
            previous_frame_sha256=self._frame_sha,
        )
        frame["frame_sha256"] = _digest(frame)
        if self._frame_sink is None and self._retained_bytes + len(
            _encoded(frame)
        ) > self._policy.get("max_evidence_bytes", 64 * 1024 * 1024):
            record["errors"].append(
                "lossless inventory retention resource budget exhausted; no further scans"
            )
            record["complete"] = False
            frame.update(
                errors=copy.deepcopy(record["errors"]),
                complete=False,
                processes=copy.deepcopy(record["processes"]),
                references=[],
            )
            frame.pop("frame_sha256")
            frame["frame_sha256"] = _digest(frame)
            self._retention_failed = True
        frame_bytes = len(_encoded(frame))
        if self._frame_sink is None:
            self._retained_bytes += frame_bytes
        self._frames.append(frame)
        self._frame_sizes.append(frame_bytes)
        self._inventory = copy.deepcopy(current)
        self._inventory_sha, self._frame_sha = frame["inventory_sha256"], frame["frame_sha256"]
        self._samples.append(copy.deepcopy(record))
        seq = self._frame_count
        self._frame_count += 1
        self._first_frame_sha = self._first_frame_sha or self._frame_sha
        if self._frame_sink is not None:
            self._canonical_stream.update(_encoded(frame) + b"\n")
            self._canonical_bytes += frame_bytes + 1
            try:
                self._check_state_budget()
                ack = self._frame_sink.append(seq, frame)
                self._check_archive_progress(ack)
                if (
                    ack.get("durable") is not True
                    or type(ack.get("seq")) is not int
                    or ack["seq"] != seq
                    or ack.get("frame_sha256") != self._frame_sha
                ):
                    raise ArchiveError("archive ACK is missing durable sequence/frame proof")
                self._archive_physical_bytes = ack["physical_bytes"]
                self._frames.clear()
                self._retained_bytes = self._state_proxy_bytes()
            except (OSError, ValueError, TypeError) as error:
                self._archive_failed, self._retention_failed = str(error), True
                raise ArchiveError(
                    f"archive failed without eviction or fallback: {error}"
                ) from error

    def _permission_descriptor(self, process, role):
        descriptor = {
            key: process[key]
            for key in (
                "pid",
                "startticks",
                "pgid",
                "sid",
                "uids",
                "comm",
                "kernel",
                "exe",
                "argv_sha256",
                "cgroup",
            )
        }
        descriptor.update(
            boot_id=self._policy["boot_id"],
            role=role["id"],
            activity=role["activity"],
            io_unavailable_reason=role.get("io_unavailable_reason"),
        )
        if "early_boot_proof" in process:
            descriptor["early_boot_proof"] = copy.deepcopy(process["early_boot_proof"])
        if role.get("identity_basis", "direct") != "direct":
            descriptor.update(
                identity_basis=role["identity_basis"],
                corroboration=copy.deepcopy(role["corroboration"]),
                exe_unavailable_reason=role["exe_unavailable_reason"],
            )
        return descriptor

    def _worker_matches(self, process):
        return self._worker is not None and all(
            process[key] == self._worker[key]
            for key in (
                "pid",
                "startticks",
                "pgid",
                "sid",
                "uids",
                "comm",
                "argv_sha256",
                "exe",
                "cgroup",
            )
        )

    def _terminal_ownership(self, process):
        """Reuse positive prior ownership for a terminal identity, without claiming reaping."""
        prior = self._owned.get((process["pid"], process["startticks"]))
        if (
            prior is None
            or process["startticks"] <= 0
            or process["state"] != "Z"
            or process["kernel"] is not False
            or any(process[key] != prior[key] for key in ("pgid", "sid", "uids", "comm", "cgroup"))
        ):
            return None
        if process["exe"] != prior["exe"] and not (
            process["exe"] is None
            and (process["exe_unavailable"] or {}).get("errno") == errno.ENOENT
        ):
            return None
        if process["argv_sha256"] not in (
            prior["argv_sha256"],
            hashlib.sha256(b"").hexdigest(),
        ) and not (
            process["argv_sha256"] is None
            and (process["argv_unavailable"] or {}).get("errno") == errno.ENOENT
        ):
            return None
        return copy.deepcopy(prior)

    def _continued_helper_ownership(self, process, record):
        """Reprove a previously observed helper by its fresh self identity after reparenting."""
        prior = self._owned.get((process["pid"], process["startticks"]))
        if (
            prior is None
            or prior.get("origin") != "helper"
            or prior.get("recipe")
            not in {recipe["id"] for recipe in self._intent.get("helpers", [])}
            or process["startticks"] <= 0
            or process["state"] == "Z"
            or process["exe"] is None
            or process["argv_sha256"] is None
            or self._baseline is None
            or not prior["boot_id"] == record["boot_id"] == self._policy["boot_id"]
            or record["pid_namespace"] != self._baseline["pid_namespace"]
            or record["pid_namespace"] != self._policy["pid_namespace"]
            or any(
                process[key] != prior[key]
                for key in ("pgid", "sid", "uids", "comm", "exe", "argv_sha256", "cgroup")
            )
        ):
            return None
        return copy.deepcopy(prior)

    def _query(self, command):
        proof = {"argv": list(command), "scan_start": self.clock()}
        self._query_records.append(proof)
        try:
            text = self.query(command)
            if not isinstance(text, str) or len(text.encode()) > 1024 * 1024:
                raise ValueError("invalid or oversized inventory query response")
            proof.update(
                bytes=len(text.encode()),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                result=text,
                complete=True,
            )
            return text
        except (OSError, ValueError, TypeError) as error:
            proof.update(complete=False, error=str(error))
            raise
        finally:
            proof["scan_end"] = self.clock()

    def _gpu_processes(self):
        pids = set()
        for row in csv.reader(io.StringIO(self._query(COMPUTE_QUERY))):
            if not row:
                continue
            if (
                len(row) != 3
                or row[0].strip() != self._policy["gpu_uuid"]
                or not row[1].strip().isdecimal()
                or int(row[1]) <= 0
                or int(row[1]) in pids
            ):
                raise ValueError("unknown or duplicate GPU inventory identity")
            pids.add(int(row[1]))
        return pids

    def _pids(self):
        pids = sorted(int(path.name) for path in self.proc_root.iterdir() if path.name.isdecimal())
        if len(pids) > 8192 or any(pid <= 0 for pid in pids):
            raise ValueError("process inventory exceeds its fixed identity bound")
        return pids

    def _container_observation(self, record):
        if self._intent["kind"] != "serving" or self._worker is None:
            return None
        expected = self._intent["server"]
        command = (
            "docker",
            "inspect",
            "--type",
            "container",
            "--format",
            DOCKER_FORMAT,
            expected["name"],
        )
        info = json.loads(self._query(command))
        if info is None:
            record["container_state"] = "absent"
            return None
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("Config"), dict)
            or not isinstance(info["Config"].get("Labels"), dict)
            or not isinstance(info.get("State"), dict)
        ):
            raise ValueError("malformed container identity metadata")
        cid = info["Id"]
        if (
            not isinstance(cid, str)
            or re.fullmatch("[0-9a-f]{64}", cid) is None
            or not isinstance(info["Image"], str)
            or re.fullmatch("sha256:[0-9a-f]{64}", info["Image"]) is None
            or (expected.get("image_id") is not None and info["Image"] != expected["image_id"])
            or info["Name"] != "/" + expected["name"]
            or info["Config"]["Image"] != expected["image"]
            or any(
                info["Config"]["Labels"].get(key) != value
                for key, value in expected["labels"].items()
            )
            or type(info["State"]["Running"]) is not bool
            or type(info["State"]["Pid"]) is not int
        ):
            raise ValueError("container name, labels, image or identity changed")
        if self._container is not None and cid != self._container["cid"]:
            raise ValueError("container identity changed; no restart ownership")
        if not info["State"]["Running"]:
            if self._container is None or info["State"]["Pid"] != 0:
                raise ValueError("unbound or inconsistent exited container identity")
            record["container_state"] = "exited"
            return None
        root = self._process(info["State"]["Pid"])
        record.setdefault("container_process_observations", []).append(copy.deepcopy(root))
        if root["exe"] is None:
            raise ValueError("container root executable identity must be directly readable")
        if not any(
            component in (cid, f"docker-{cid}.scope") for component in root["cgroup"].split("/")
        ):
            raise ValueError("container root cgroup does not bind the full CID")
        self._qualify_io(root, self._intent.get("io_unavailable_reason"))
        binding = {
            "cid": cid,
            "name": expected["name"],
            "image": info["Image"],
            "image_ref": info["Config"]["Image"],
            "pid": root["pid"],
            "ppid": root["ppid"],
            "startticks": root["startticks"],
            "cgroup": root["cgroup"],
            "boot_id": self._policy["boot_id"],
        }
        if self._container is not None and binding != self._container:
            raise ValueError("container engine identity or source changed")
        self._container = copy.deepcopy(binding)
        record["container_state"] = "running"
        return binding

    @staticmethod
    def _container_matches(process, binding):
        return (
            binding is not None
            and (
                process["cgroup"] == binding["cgroup"]
                or process["cgroup"].startswith(binding["cgroup"] + "/")
            )
            and (process["pid"] != binding["pid"] or process["startticks"] == binding["startticks"])
        )

    def _read(self, path, *, limit=65536):
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"oversized process metadata: {path}")
        return data

    def _trusted_file(self, proof, *, unit=False):
        cache_key = _digest({"file": proof, "unit": unit})
        if cache_key in self._proof_cache:
            return
        path = Path(proof["path"])
        if str(path.resolve(strict=True)) != proof["path"]:
            raise ValueError("OS proof file is not its canonical direct path")

        def observed(target, *, regular=False):
            metadata = self._file_stat(target, follow_symlinks=False)
            result = {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "uid": metadata.st_uid,
                "mode": metadata.st_mode,
            }
            if regular:
                result.update(size=metadata.st_size, mtime_ns=metadata.st_mtime_ns)
            return result

        for entry in proof["ancestors"]:
            actual = observed(Path(entry["path"]))
            if (
                actual != entry["stat"]
                or actual["uid"] != 0
                or actual["mode"] & 0o022
                or not stat.S_ISDIR(actual["mode"])
            ):
                raise ValueError("OS proof ancestor identity or protection changed")
        actual = observed(path, regular=True)
        if (
            actual != proof["stat"]
            or actual["uid"] != 0
            or actual["mode"] & 0o022
            or not stat.S_ISREG(actual["mode"])
        ):
            raise ValueError("OS proof configured file identity or protection changed")
        if unit:
            data = self._read(path)
            if len(data) != actual["size"] or hashlib.sha256(data).hexdigest() != proof["sha256"]:
                raise ValueError("OS unit source hash changed")
            if observed(path, regular=True) != actual:
                raise ValueError("OS unit identity changed during its bounded read")
        self._proof_cache[cache_key] = True

    def _trusted_alias(self, alias):
        entries = {entry["path"]: entry for entry in alias["components"]}
        if len(entries) != len(alias["components"]):
            raise ValueError("configured alias proof repeats a component")
        pending, current, seen = deque(Path(alias["path"]).parts[1:]), Path("/"), set()

        def check(path):
            expected = entries.get(str(path))
            if expected is None:
                raise ValueError("configured alias proof omits a traversed component")
            metadata = self._file_stat(path, follow_symlinks=False)
            actual = {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "uid": metadata.st_uid,
                "mode": metadata.st_mode,
            }
            if actual != expected["stat"] or actual["uid"] != 0:
                raise ValueError("configured alias component identity changed")
            if not stat.S_ISLNK(actual["mode"]) and actual["mode"] & 0o022:
                raise ValueError("configured alias traverses a writable component")
            seen.add(str(path))
            return expected, actual["mode"]

        _, root_mode = check(current)
        if not stat.S_ISDIR(root_mode):
            raise ValueError("configured alias has no protected root directory")
        steps = 0
        while pending:
            steps += 1
            if steps > 128:
                raise ValueError("configured alias resolution exceeds its fixed bound")
            part = pending.popleft()
            if part in ("", "."):
                continue
            if part == "..":
                current = current.parent
                continue
            candidate = current / part
            expected, mode = check(candidate)
            if stat.S_ISLNK(mode):
                target = os.readlink(candidate)
                if not isinstance(expected.get("link"), str) or target != expected["link"]:
                    raise ValueError("configured alias link target changed")
                target_path = Path(target)
                if target_path.is_absolute():
                    current = Path("/")
                    pending.extendleft(reversed(target_path.parts[1:]))
                else:
                    pending.extendleft(reversed(target_path.parts))
            else:
                if pending and not stat.S_ISDIR(mode):
                    raise ValueError("configured alias traverses a non-directory")
                current = candidate
        if str(current) != alias["target"] or seen != entries.keys():
            raise ValueError("configured alias target or complete traversal proof changed")
        return alias["path"]

    def _verify_os_role(self, process, role, visited=()):
        if role["id"] in visited:
            raise ValueError("cyclic OS parent proof")
        proof = role["corroboration"]
        if role["identity_basis"] == "baseline-terminal-z":
            key = process["pid"], process["startticks"]
            if (
                process["state"] != "Z"
                or process["cpu_ticks"] != proof["cpu_ticks"]
                or process["gpu"]
                or (self._baseline is not None and key not in self._permitted)
            ):
                raise ValueError("terminal identity is not the pre-existing frozen non-GPU task")
            process["corroboration"] = {
                "identity_basis": role["identity_basis"],
                "decision_sha256": OS_CONTEXT_DECISION,
                "proof_sha256": _digest(proof),
            }
            return
        executable = proof["configured_executable"]
        self._trusted_file(executable)
        configured_path = executable["path"]
        if "configured_alias" in proof:
            configured_path = self._trusted_alias(proof["configured_alias"])
        for configured in proof.get("configured_inputs", []):
            self._trusted_file(configured)
        if process["exe"] is None:
            if not process["exe_unavailable"] or process["exe_unavailable"]["errno"] not in (
                errno.EACCES,
                errno.EPERM,
            ):
                raise ValueError("OS role executable has no accepted visibility proof")
            process["exe_visibility"] = {
                "status": "corroborated-unavailable",
                "reason": role["exe_unavailable_reason"],
            }
        elif process["exe"] != executable["path"]:
            raise ValueError("readable OS executable differs from its configured identity")
        else:
            process["exe_visibility"] = {"status": "directly-observed"}
        if role["identity_basis"] == "systemd-main":
            unit = proof["unit"]
            self._trusted_file(unit["file"], unit=True)
            cache_key = _digest({"unit": unit, "configured_executable": executable["path"]})
            if cache_key not in self._proof_cache:
                text = self._query((*SYSTEMD_PREFIX, unit["name"]))
                lines = text.splitlines()
                pairs = [line.split("=", 1) for line in lines]
                if (
                    any(len(pair) != 2 for pair in pairs)
                    or len(pairs) != 4
                    or len({pair[0] for pair in pairs}) != 4
                ):
                    raise ValueError("ambiguous systemd unit metadata")
                metadata = dict(pairs)
                if (
                    set(metadata) != {"MainPID", "ControlGroup", "FragmentPath", "ExecStart"}
                    or not metadata["MainPID"].isdecimal()
                ):
                    raise ValueError("unavailable typed systemd unit identity")
                metadata["MainPID"] = int(metadata["MainPID"])
                if (
                    _digest(metadata) != unit["metadata_sha256"]
                    or _digest(metadata) != _digest(unit["metadata"])
                    or metadata["MainPID"] != process["pid"]
                    or not (
                        metadata["ControlGroup"] == process["cgroup"]
                        or (
                            role.get("main_subgroup") is True
                            and process["cgroup"].startswith(
                                metadata["ControlGroup"].rstrip("/") + "/"
                            )
                        )
                    )
                    or metadata["FragmentPath"] != unit["file"]["path"]
                    or re.findall(r"\{\s*path=([^ ;{}]+)\s*;", metadata["ExecStart"])
                    != [configured_path]
                ):
                    raise ValueError("systemd main PID, unit or configured executable changed")
                self._proof_cache[cache_key] = True
        process["corroboration"] = {
            "identity_basis": role["identity_basis"],
            "decision_sha256": self._policy["decision_sha256"],
            "proof_sha256": _digest(proof),
        }
        if role["identity_basis"] == "systemd-main":
            process["corroboration"]["root_unit"] = proof["unit"]["name"]
        if role["identity_basis"] == "os-containerd-shim":
            binding = self._container
            daemon = next(
                (
                    row
                    for row in self._scan_processes
                    if (self._role(row) or {}).get("id") == role["daemon_role"]
                ),
                None,
            )
            daemon_role = None if daemon is None else self._role(daemon)
            if (
                binding is None
                or daemon_role is None
                or daemon_role.get("identity_basis") != "systemd-main"
                or (process.get("present_at_end", True) and not daemon.get("present_at_end", True))
                or daemon_role["corroboration"]["unit"]["name"] != "containerd.service"
            ):
                raise ValueError("shim lacks its registered CID or live trusted containerd MAIN")
            self._verify_os_role(daemon, daemon_role, (*visited, role["id"]))
            init = next(
                (
                    row
                    for row in self._scan_processes
                    if row["pid"] == binding["pid"] and row["startticks"] == binding["startticks"]
                ),
                None,
            )
            if (
                init is None
                or binding["ppid"] != process["pid"]
                or init["ppid"] != process["pid"]
                or init["startticks"] < process["startticks"]
                or not self._container_matches(init, binding)
                or process["ppid"] != 1
                or process["pgid"] != process["pid"]
                or process["sid"] != daemon["sid"]
            ):
                raise ValueError("registered CID/init/shim independent correlation is unavailable")
            process["corroboration"].update(
                decision_sha256=OS_CONTEXT_DECISION,
                cid=binding["cid"],
                daemon={"pid": daemon["pid"], "startticks": daemon["startticks"]},
                init={"pid": init["pid"], "startticks": init["startticks"], "ppid": init["ppid"]},
            )
        if role["identity_basis"] in ("os-fixed-helper", "os-telemetry-helper"):
            expected = role.get("parent_identity")
            parent_pid = process["ppid"] if expected is None else expected["pid"]
            parent = next((row for row in self._scan_processes if row["pid"] == parent_pid), None)
            parent_role = None if parent is None else self._role(parent)
            if (
                parent is None
                or parent_role is None
                or parent_role["id"] != role["parent_role"]
                or (
                    expected is not None
                    and any(parent[key] != value for key, value in expected.items())
                )
                or (process.get("present_at_end", True) and not parent.get("present_at_end", True))
                or process["ppid"] != parent["pid"]
                or process["startticks"] < parent["startticks"]
            ):
                raise ValueError("fixed OS child lacks its exact live approved parent")
            if parent_role.get("identity_basis", "direct") != "direct":
                self._verify_os_role(parent, parent_role, (*visited, role["id"]))
            process["corroboration"].update(
                decision_sha256=OS_CONTEXT_DECISION,
                parent={"pid": parent["pid"], "startticks": parent["startticks"]},
                root_unit=parent.get("corroboration", {}).get("root_unit"),
            )
            if role["identity_basis"] == "os-telemetry-helper":
                group = (
                    (process["pid"], process["pid"])
                    if role["session"] == "new"
                    else (parent["pgid"], parent["sid"])
                )
                peers = [
                    row
                    for row in self._scan_processes
                    if row["ppid"] == parent["pid"]
                    and (self._role(row) or {}).get("id") == role["id"]
                ]
                if (
                    (process["pgid"], process["sid"]) != group
                    or process["corroboration"]["root_unit"] != role["root_unit"]
                    or len(peers) > role["max_live"]
                ):
                    raise ValueError("telemetry root, session or concurrency proof changed")

    def _executable(self, directory, kernel, terminal=False):
        if kernel is True:
            return None, None
        try:
            return os.readlink(directory / "exe"), None
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM) and not (
                terminal and error.errno == errno.ENOENT
            ):
                raise ValueError("unclassified executable visibility error") from error
            return None, {"errno": error.errno, "error": str(error)}

    def _argv_metadata(self, directory, terminal):
        try:
            return hashlib.sha256(self._read(directory / "cmdline")).hexdigest(), None
        except OSError as error:
            if not terminal or error.errno not in (errno.EACCES, errno.EPERM, errno.ENOENT):
                raise
            return None, {"errno": error.errno, "error": str(error)}

    def _stat(self, pid):
        text = self._read(self.proc_root / str(pid) / "stat").decode()
        head, separator, tail = text.rpartition(")")
        fields = tail.split()
        if not separator or len(fields) < 20 or int(head.split("(", 1)[0]) != pid:
            raise ValueError(f"invalid process stat identity: {pid}")
        result = {
            "pid": pid,
            "comm": head.split("(", 1)[1],
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "startticks": int(fields[19]),
        }
        if result["state"] not in set("RSDZTtXxKWPI"):
            raise ValueError(f"unknown process state: {pid}")
        if (
            pid <= 0
            or result["startticks"] < 0
            or result["ppid"] < 0
            or result["pgid"] < 0
            or result["sid"] < 0
            or result["cpu_ticks"] < 0
        ):
            raise ValueError(f"invalid process identity counters: {pid}")
        return result

    def _process(self, pid):
        self._process_partial = {"pid": pid}
        result = self._stat(pid)
        self._process_partial = result
        _observe_running_state(result, result["state"])
        terminal = result["state"] == "Z" and (
            (pid, result["startticks"]) in self._owned
            or any(
                role.get("identity_basis") == "baseline-terminal-z"
                and role["corroboration"]["pid"] == pid
                and role["corroboration"]["startticks"] == result["startticks"]
                for role in self._policy["roles"]
            )
        )
        directory = self.proc_root / str(pid)
        status = dict(
            line.split(":", 1)
            for line in self._read(directory / "status").decode().splitlines()
            if ":" in line
        )
        kernel = status.get("Kthread", "").strip()
        if kernel not in ("", "0", "1"):
            raise ValueError(f"invalid kernel-thread classification: {pid}")
        result["kernel"] = {"": None, "0": False, "1": True}[kernel]
        uids = status.get("Uid", "").split()
        if len(uids) != 4 or any(not uid.isdecimal() for uid in uids):
            raise ValueError(f"unknown full UID identity: {pid}")
        result["uids"] = [int(uid) for uid in uids]
        if kernel != "1" and (result["pgid"] == 0 or result["sid"] == 0):
            raise ValueError(f"nonkernel process has unknown group or session: {pid}")
        result["exe"], result["exe_unavailable"] = self._executable(
            directory, result["kernel"], terminal
        )
        result["argv_sha256"], result["argv_unavailable"] = self._argv_metadata(directory, terminal)
        groups = self._read(directory / "cgroup").decode().splitlines()
        unified = [line[3:] for line in groups if line.startswith("0::")]
        if len(unified) != 1 or not unified[0].startswith("/"):
            raise ValueError(f"unknown unified cgroup identity: {pid}")
        result["cgroup"] = unified[0]
        result["io"] = None
        result["io_unavailable"] = None
        if kernel != "1":
            try:
                counters = dict(
                    line.split(":", 1)
                    for line in self._read(directory / "io").decode().splitlines()
                    if ":" in line
                )
                result["io"] = {
                    key: int(counters[key])
                    for key in ("rchar", "wchar", "read_bytes", "write_bytes")
                }
                if any(value < 0 for value in result["io"].values()):
                    raise ValueError(f"invalid process I/O counters: {pid}")
            except OSError as error:
                if not isinstance(error, PermissionError) and not (
                    terminal and error.errno == errno.ENOENT
                ):
                    raise
                result["io_unavailable"] = str(error)
        if (
            (result["exe"], result["exe_unavailable"])
            != self._executable(directory, result["kernel"], terminal)
            or (result["argv_sha256"], result["argv_unavailable"])
            != self._argv_metadata(directory, terminal)
            or self._read(directory / "cgroup").decode().splitlines() != groups
        ):
            raise ValueError(f"process metadata changed during inventory: {pid}")
        end = self._stat(pid)
        end_status = dict(
            line.split(":", 1)
            for line in self._read(directory / "status").decode().splitlines()
            if ":" in line
        )
        if (
            end_status.get("Uid", "").split() != uids
            or end_status.get("Kthread", "").strip() != kernel
        ):
            raise ValueError(f"process UID or kernel identity changed during inventory: {pid}")
        if any(end[key] != result[key] for key in ("startticks", "pgid", "sid", "comm")):
            raise ValueError(f"process identity changed during inventory: {pid}")
        if end["cpu_ticks"] < result["cpu_ticks"]:
            raise ValueError(f"process counters regressed during inventory: {pid}")
        _observe_running_state(result, end["state"])
        result.update(end)
        previous = self._previous.get((pid, result["startticks"]))
        result["cpu_delta"] = (
            None if previous is None else result["cpu_ticks"] - previous["cpu_ticks"]
        )
        result["io_delta"] = None
        if previous is not None and result["io"] is not None and previous["io"] is not None:
            result["io_delta"] = {
                key: result["io"][key] - previous["io"][key] for key in result["io"]
            }
        deltas = [] if result["cpu_delta"] is None else [result["cpu_delta"]]
        deltas += list((result["io_delta"] or {}).values())
        if any(delta < 0 for delta in deltas):
            raise ValueError(f"process counters regressed: {pid}")
        result["active"] = (
            previous is None or bool(result["observed_running_states"]) or any(deltas)
        )
        result["active"] = bool(result["active"])
        if len(_encoded(result)) > 16384:
            raise ValueError(f"process metadata exceeds the retention row bound: {pid}")
        return result

    @staticmethod
    def _qualify_io(process, reason=None):
        if process["io"] is not None:
            process["io_visibility"] = {"status": "available"}
        elif process["kernel"] is True:
            process["io_visibility"] = {
                "status": "not-applicable",
                "reason": "verified kernel thread",
            }
        elif process["io_unavailable"] and isinstance(reason, str) and reason.strip():
            process["io_visibility"] = {"status": "declared-not-required", "reason": reason}
            process["active"] = True
            process["activity_basis"] = "conservative activity: I/O unknown"
        else:
            raise ValueError(f"required I/O activity counters unavailable: {process['pid']}")

    def _classify(self, record, container, gpu_pids):
        processes = record["processes"]
        self._scan_processes = processes
        resolved = {}
        for process in processes:
            terminal = self._terminal_ownership(process)
            if terminal is not None:
                process["gpu"] = process["pid"] in gpu_pids
                process["classification"] = "owned-terminal-" + terminal["origin"]
                process["ownership"] = terminal
                process["owned_terminal"] = {"requires_reap": True}
                continue
            helper = self._continued_helper_ownership(process, record)
            if helper is not None:
                process["gpu"] = process["pid"] in gpu_pids
                process["classification"] = "owned-helper"
                process["ownership"] = helper
                resolved[helper["recipe"]] = process
                continue
            origin = (
                "worker"
                if process["exe"] is not None and self._worker_matches(process)
                else "container"
                if process["exe"] is not None and self._container_matches(process, container)
                else None
            )
            process["gpu"] = process["pid"] in gpu_pids
            process["classification"] = "unknown" if origin is None else "owned-" + origin
            if origin:
                proof = {
                    key: process[key]
                    for key in (
                        "pid",
                        "startticks",
                        "pgid",
                        "sid",
                        "uids",
                        "comm",
                        "exe",
                        "argv_sha256",
                        "cgroup",
                    )
                }
                proof.update(boot_id=self._policy["boot_id"], origin=origin)
                if origin == "container":
                    proof["cid"] = container["cid"]
                else:
                    resolved["worker"] = process
                process["ownership"] = proof
        pending = list(self._intent.get("helpers", []))
        while pending:
            progress = False
            for recipe in pending[:]:
                parent = resolved.get(recipe["parent"])
                if parent is None:
                    continue
                candidates = [
                    process
                    for process in processes
                    if process["classification"] == "unknown"
                    and process["ppid"] == parent["pid"]
                    and (
                        not process.get("present_at_end", True)
                        or parent.get("present_at_end", True)
                    )
                    and process["startticks"] >= parent["startticks"]
                    and process["exe"] == recipe["exe"]
                    and process["argv_sha256"] == recipe["argv_sha256"]
                    and process["uids"] == recipe.get("uids", parent["uids"])
                    and process["cgroup"] == recipe.get("cgroup", self._worker["cgroup"])
                    and (process["pgid"], process["sid"])
                    == (
                        (process["pid"], process["pid"])
                        if recipe.get("session", "inherit") == "new"
                        else (parent["pgid"], parent["sid"])
                    )
                ]
                if len(candidates) == 1 and recipe["id"] not in resolved:
                    process = candidates[0]
                    process["classification"] = "owned-helper"
                    process["ownership"] = {
                        key: process[key]
                        for key in (
                            "pid",
                            "startticks",
                            "pgid",
                            "sid",
                            "uids",
                            "comm",
                            "exe",
                            "argv_sha256",
                            "cgroup",
                        )
                    }
                    process["ownership"].update(
                        boot_id=self._policy["boot_id"],
                        origin="helper",
                        recipe=recipe["id"],
                        parent={"pid": parent["pid"], "startticks": parent["startticks"]},
                    )
                    resolved[recipe["id"]] = process
                elif candidates:
                    record["errors"].append(f"duplicate helper recipe identities: {recipe['id']}")
                pending.remove(recipe)
                progress = True
            if not progress:
                break
        profiles = {} if self._worker is None else {"worker": self._worker["uids"]}
        unresolved = list(self._intent.get("helpers", []))
        while unresolved and profiles:
            progress = False
            for recipe in unresolved[:]:
                if recipe["parent"] in profiles:
                    profiles[recipe["id"]] = recipe.get("uids", profiles[recipe["parent"]])
                    unresolved.remove(recipe)
                    progress = True
            if not progress:
                break
        for process in processes:
            errors_before = len(record["errors"])
            pid, start = process["pid"], process["startticks"]
            owned = process["classification"].startswith("owned-")
            role = self._role(process)
            if process["startticks"] == 0:
                if (
                    role
                    and role.get("early_boot") is True
                    and self._policy.get("context_decision_sha256") == OS_CONTEXT_DECISION
                ):
                    process["early_boot_proof"] = {
                        "decision_sha256": OS_CONTEXT_DECISION,
                        "role": role["id"],
                        "measured": True,
                        "boot_id": self._policy["boot_id"],
                        "pid_namespace": self._policy["pid_namespace"],
                        "pid": pid,
                        "startticks": 0,
                    }
                else:
                    record["errors"].append(
                        f"measured zero start lacks approved early-boot proof: {pid}"
                    )
                    record["ownership_complete"] = False
            if owned:
                self._owned[(pid, start)] = copy.deepcopy(process["ownership"])
            elif role is None:
                record["errors"].append(f"unclassified process identity: {pid}")
            else:
                process["classification"] = role["id"]
                context_verified = False
                if role.get("identity_basis", "direct") != "direct":
                    try:
                        self._verify_os_role(process, role)
                        context_verified = True
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        record["errors"].append(
                            f"OS identity corroboration unavailable: {pid}: {error}"
                        )
                        record["ownership_complete"] = False
                descriptor = self._permission_descriptor(process, role)
                if self._baseline is not None and (pid, start) not in self._permitted:
                    if (
                        process["kernel"] is True
                        and role.get("allow_new_kernel") is True
                        and role["activity"] == "permitted"
                    ) or (
                        role.get("identity_basis") in ("os-telemetry-helper", "os-containerd-shim")
                        and context_verified
                    ):
                        self._permitted[(pid, start)] = descriptor
                    else:
                        record["errors"].append(
                            f"new identity has no frozen role permission: {pid}"
                        )
                elif self._baseline is not None and descriptor != self._permitted[(pid, start)]:
                    record["errors"].append(
                        f"process descriptor differs from frozen permission: {pid}"
                    )
                if (
                    role["activity"] == "idle"
                    and process["cpu_delta"] is not None
                    and process["active"]
                ):
                    record["errors"].append(f"idle role observed activity: {pid}")
                elif role["activity"] not in ("idle", "permitted"):
                    record["errors"].append(f"unknown role activity permission: {pid}")
            reason = (
                "Previously proved owned terminal identity; reaping remains required."
                if "owned_terminal" in process
                else self._intent.get("io_unavailable_reason")
                if owned
                else role.get("io_unavailable_reason")
                if role and role["activity"] == "permitted"
                else None
            )
            try:
                self._qualify_io(process, reason)
            except ValueError as error:
                record["errors"].append(str(error))
            if process["gpu"] and not owned:
                record["errors"].append(f"GPU process {pid} has no proven ownership identity")
            if (
                not owned
                and self._worker is not None
                and process["startticks"] >= self._worker["startticks"]
            ):
                possible = [
                    recipe["id"]
                    for recipe in self._intent.get("helpers", [])
                    if process["exe"] == recipe["exe"]
                    and process["argv_sha256"] == recipe["argv_sha256"]
                    and process["cgroup"] == recipe.get("cgroup", self._worker["cgroup"])
                    and process["uids"] == profiles.get(recipe["id"])
                ]
                if possible:
                    process["ownership_uncertain"] = {
                        "possible_helper_recipes": possible,
                        "reason": "current parent identity unavailable",
                    }
                    record["errors"].append(
                        f"possible owned helper lacks current parent proof: {pid}"
                    )
                    record["ownership_complete"] = False
            if (
                process["exe"] is None
                and process["kernel"] is not True
                and "corroboration" not in process
                and "owned_terminal" not in process
            ):
                record["errors"].append(f"required executable identity unavailable: {pid}")
                record["ownership_complete"] = False
            process["identity_qualified"] = len(record["errors"]) == errors_before

    def _role(self, process):
        for role in self._policy["roles"]:
            if role.get("identity_basis", "direct") != "direct":
                expected = role["corroboration"]
                dynamic = role.get("identity_basis") in (
                    "os-telemetry-helper",
                    "os-containerd-shim",
                )
                keys = (
                    ("uids", "comm", "cgroup")
                    if dynamic
                    else (
                        "pid",
                        "startticks",
                        "pgid",
                        "sid",
                        "uids",
                        "comm",
                        "argv_sha256",
                        "cgroup",
                    )
                )
                matches = all(process[key] == expected[key] for key in keys)
                if dynamic:
                    sources = {
                        "parent.pid": str(process["ppid"]),
                        "registered.cid": None
                        if self._container is None
                        else self._container["cid"],
                    }
                    argv = [
                        sources[token["source"]] if isinstance(token, dict) else token
                        for token in expected["argv"]
                    ]
                    matches = (
                        matches
                        and None not in argv
                        and process["argv_sha256"]
                        == hashlib.sha256(("\0".join(argv) + "\0").encode()).hexdigest()
                    )
            elif role.get("kernel") is True:
                matches = process["kernel"] is True
            else:
                matches = (
                    all(
                        process.get(key) == role.get(key)
                        for key in ("exe", "argv_sha256", "cgroup")
                    )
                    and role.get("exe") is not None
                )
            identities = role.get("identities")
            if matches and (
                identities is None
                or {"pid": process["pid"], "startticks": process["startticks"]} in identities
            ):
                return role
        return None

    def _finish_inventory(self, record, first_pids, first_gpu):
        """One bounded closing view; retain failures and reconcile only fully proved transitions."""
        record["boundary_observations"]["start"] = copy.deepcopy(record["processes"])
        earlier_container = len(record.get("container_process_observations", []))
        self._container_observation(record)
        record["container_binding"] = copy.deepcopy(self._container)
        last_gpu, last_pids = self._gpu_processes(), self._pids()
        union = {}

        def add(row):
            key = _identity(row)
            old = union.get(key)
            if old is not None:
                if any(
                    old[field] != row[field]
                    for field in (
                        "pgid",
                        "sid",
                        "uids",
                        "comm",
                        "exe",
                        "argv_sha256",
                        "cgroup",
                        "kernel",
                    )
                ):
                    record["errors"].append(
                        f"process metadata changed between observations: {row['pid']}"
                    )
                    record["ownership_complete"] = False
                    record["races"].append(
                        {"pid": row["pid"], "classification": "metadata-changed"}
                    )
                if row["cpu_ticks"] < old["cpu_ticks"]:
                    record["errors"].append(
                        f"process counters regressed between observations: {row['pid']}"
                    )
                row["active"] = bool(row["active"] or old["active"])
                row["observed_running_states"] = sorted(
                    set(row["observed_running_states"]) | set(old["observed_running_states"])
                )
            union[key] = row

        container_rows = record.get("container_process_observations", [])
        for row in container_rows[:earlier_container]:
            add(copy.deepcopy(row))
        for row in record["processes"]:
            add(row)
        for row in container_rows[earlier_container:]:
            add(copy.deepcopy(row))
        added = []
        for pid in sorted(set(last_pids) - set(first_pids)):
            try:
                row = self._process(pid)
                added.append(copy.deepcopy(row))
                add(row)
            except (OSError, ValueError, KeyError) as error:
                record["errors"].append(f"closing process identity {pid}: {error}")
                record["ownership_complete"] = False
                record["races"].append({"pid": pid, "classification": "unresolved-birth"})
                record["read_failures"].append(
                    {
                        "pid": pid,
                        "error": str(error),
                        "partial": copy.deepcopy(self._process_partial),
                    }
                )
        final_stats = {}
        for pid in last_pids:
            try:
                final_stats[pid] = self._stat(pid)
                key = _identity(final_stats[pid])
                row = union.get(key)
                if row is None:
                    record["errors"].append(
                        f"closing PID identity has no complete observed descriptor: {pid}"
                    )
                    record["ownership_complete"] = False
                    record["races"].append(
                        {"pid": pid, "classification": "PID-reuse-or-unresolved"}
                    )
                    continue
                latest = final_stats[pid]
                if any(
                    latest[field] != row[field] for field in ("startticks", "pgid", "sid", "comm")
                ):
                    record["errors"].append(f"closing process identity changed: {pid}")
                    record["ownership_complete"] = False
                if latest["cpu_ticks"] < row["cpu_ticks"]:
                    record["errors"].append(f"closing process counters regressed: {pid}")
                    record["ownership_complete"] = False
                _observe_running_state(row, latest["state"])
                row.update(latest)
                previous = self._previous.get((pid, row["startticks"]))
                row["cpu_delta"] = (
                    None if previous is None else row["cpu_ticks"] - previous["cpu_ticks"]
                )
                row["active"] = bool(
                    row["active"] or row["state"] in ("R", "D") or row["cpu_delta"]
                )
            except (OSError, ValueError, KeyError) as error:
                record["errors"].append(f"closing stat identity {pid}: {error}")
                record["ownership_complete"] = False
                record["races"].append(
                    {"pid": pid, "classification": "unresolved-closing-identity"}
                )
        for row in union.values():
            final = final_stats.get(row["pid"])
            row["present_at_end"] = final is not None and final["startticks"] == row["startticks"]
        record["processes"] = sorted(
            union.values(), key=lambda row: (row["pid"], row["startticks"])
        )
        record["boundary_observations"]["end"] = {
            "additional_full_rows": added,
            "stats": copy.deepcopy(list(final_stats.values())),
        }
        record["boundaries"]["end"] = {
            "pids": last_pids,
            "stats": copy.deepcopy(list(final_stats.values())),
            "compute_pids": sorted(last_gpu),
            "at": self.clock(),
        }
        self._classify(record, self._container, first_gpu | last_gpu)
        by_pid = {}
        for row in record["processes"]:
            by_pid.setdefault(row["pid"], []).append(row)
        for pid in (first_gpu | last_gpu) - by_pid.keys():
            record["errors"].append(f"GPU process identity unavailable in observed union: {pid}")
            record["ownership_complete"] = False
        enabled = self._policy.get("context_decision_sha256") == OS_CONTEXT_DECISION
        changes = [
            ("birth", set(last_pids) - set(first_pids)),
            ("exit", set(first_pids) - set(last_pids)),
            ("birth-and-exit", by_pid.keys() - set(first_pids) - set(last_pids)),
            ("compute-enter", last_gpu - first_gpu),
            ("compute-exit", first_gpu - last_gpu),
        ]
        for kind, pids in changes:
            for pid in sorted(pids):
                rows = by_pid.get(pid, [])
                row = rows[0] if len(rows) == 1 else None
                proof = None if row is None else row.get("ownership")
                if proof is None and row is not None:
                    role = self._role(row)
                    if role is not None and not kind.startswith("compute-"):
                        if role["activity"] == "idle" and kind in ("exit", "birth-and-exit"):
                            row["active"] = True
                            row["activity_uncertain"] = (
                                "idle-only exit lacks final activity counters"
                            )
                            row["identity_qualified"] = False
                        else:
                            proof = self._permission_descriptor(row, role)
                if (
                    not enabled
                    or row is None
                    or not row.get("identity_qualified")
                    or proof is None
                    or (
                        kind.startswith("compute-")
                        and not row["classification"].startswith("owned-")
                    )
                ):
                    record["errors"].append(f"unproved process lifecycle transition {kind}: {pid}")
                    record["ownership_complete"] = False
                    record["races"].append({"pid": pid, "classification": "unresolved-" + kind})
                else:
                    record["resolved_lifecycle"].append(
                        {
                            "kind": kind,
                            "pid": pid,
                            "startticks": row["startticks"],
                            "proof": copy.deepcopy(proof),
                            "decision_sha256": OS_CONTEXT_DECISION,
                            "from_boundary": "start",
                            "to_boundary": "end",
                        }
                    )

    def sample(self):
        if self._archive_metadata is not None:
            raise ArchiveError("archive is closed; no scans after finish")
        if self._retention_failed:
            raise ValueError(
                "lossless inventory retention exhausted; a reset cannot qualify this interval"
            )
        started = self.clock()
        self._proof_cache = {}
        self._query_records = []
        record = {
            "at": started,
            "scan_start": started,
            "complete": True,
            "errors": [],
            "races": [],
            "processes": [],
            "boot_id": None,
            "pid_namespace": None,
            "ownership_complete": True,
            "queries": self._query_records,
            "boundaries": {"start": {}, "end": {}},
            "boundary_observations": {"start": [], "end": {}},
            "resolved_lifecycle": [],
            "read_failures": [],
        }
        try:
            if type(started) not in (int, float) or not math.isfinite(started) or started < 0:
                raise ValueError("invalid inventory clock identity")
            if self._samples and started < self._samples[-1]["at"]:
                raise ValueError("inventory clock moved backwards between samples")
            boot = self._read(self.proc_root / "sys/kernel/random/boot_id").decode().strip()
            namespace = os.readlink(self.proc_root / "self/ns/pid")
            if boot != self._policy["boot_id"] or namespace != self._policy["pid_namespace"]:
                raise ValueError("host boot or PID namespace identity changed")
            record.update(boot_id=boot, pid_namespace=namespace)
            if self._query(GPU_QUERY).splitlines() != [self._policy["gpu_uuid"]]:
                raise ValueError("GPU inventory identity changed")
            record["gpu_uuid"] = self._policy["gpu_uuid"]
            pids = self._pids()
            record["boundaries"]["start"] = {"pids": pids, "stats": [], "at": self.clock()}
            initial = {}
            for pid in pids:
                try:
                    initial[pid] = self._stat(pid)
                    record["boundaries"]["start"]["stats"].append(copy.deepcopy(initial[pid]))
                except (OSError, ValueError, KeyError) as error:
                    record["errors"].append(f"initial process identity {pid}: {error}")
                    record["ownership_complete"] = False
                    record["races"].append(
                        {"pid": pid, "classification": "unresolved-initial-identity"}
                    )
            self._container_observation(record)
            record["container_binding"] = copy.deepcopy(self._container)
            gpu_pids = self._gpu_processes()
            record["boundaries"]["start"]["compute_pids"] = sorted(gpu_pids)
            for pid in pids:
                try:
                    process = self._process(pid)
                    if pid in initial and any(
                        process[key] != initial[pid][key] for key in ("startticks", "pgid", "sid")
                    ):
                        raise ValueError("process identity changed across the inventory boundary")
                    if pid in initial and process["cpu_ticks"] < initial[pid]["cpu_ticks"]:
                        raise ValueError("process CPU counters regressed from the first boundary")
                    if pid in initial:
                        _observe_running_state(process, initial[pid]["state"])
                    record["processes"].append(process)
                except (OSError, ValueError, KeyError) as error:
                    record["errors"].append(f"process identity {pid}: {error}")
                    record["ownership_complete"] = False
                    record["races"].append({"pid": pid, "classification": "unresolved-identity"})
                    record["read_failures"].append(
                        {
                            "pid": pid,
                            "error": str(error),
                            "partial": copy.deepcopy(self._process_partial),
                        }
                    )
            self._finish_inventory(record, pids, gpu_pids)
            end_boot = self._read(self.proc_root / "sys/kernel/random/boot_id").decode().strip()
            end_namespace = os.readlink(self.proc_root / "self/ns/pid")
            record["host_boundary"] = {"boot_id": end_boot, "pid_namespace": end_namespace}
            if end_boot != boot or end_namespace != namespace:
                record["errors"].append("boot or PID namespace changed during the inventory")
                record["races"].append({"classification": "host-identity-changed"})
                record["ownership_complete"] = False
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            record["errors"].append(str(error))
            record["ownership_complete"] = False
        record["scan_end"] = record["at"] = self.clock()
        if (
            type(started) not in (int, float)
            or type(record["at"]) not in (int, float)
            or not math.isfinite(started)
            or not math.isfinite(record["at"])
            or started < 0
            or record["at"] < 0
            or not 0 <= record["at"] - started <= 5
        ):
            record["errors"].append("invalid or overlong process inventory interval")
            record["ownership_complete"] = False
        for observation in [record, *record["queries"], *record["boundaries"].values()]:
            for key in ("at", "scan_start", "scan_end"):
                if key in observation and (
                    type(observation[key]) not in (int, float)
                    or not math.isfinite(observation[key])
                    or observation[key] < 0
                ):
                    observation.setdefault("invalid_clock", {})[key] = repr(observation[key])[:128]
                    observation[key] = None
                    record["errors"].append("unknown inventory query or scan clock boundary")
                    record["ownership_complete"] = False
        for query in record["queries"]:
            if (
                not query.get("invalid_clock")
                and record["scan_start"] is not None
                and record["scan_end"] is not None
                and not record["scan_start"]
                <= query["scan_start"]
                <= query["scan_end"]
                <= record["scan_end"]
            ):
                record["errors"].append("inventory query clock interval escaped the scan boundary")
                record["ownership_complete"] = False
        record["complete"] = not record["errors"]
        self._previous = {
            (process["pid"], process["startticks"]): process
            for process in record["processes"]
            if process.get("present_at_end", True)
        }
        self._retain(record)
        self._errors.extend(record["errors"])
        self._violations.extend(record["errors"])
        return copy.deepcopy(record)
