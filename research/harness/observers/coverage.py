from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from research.harness.models import RunResult, ScenarioSpec


@dataclass(slots=True)
class CoverageObserverConfig:
    k_sections: int = 1000
    layer_name: str | None = None
    profile_path: Path | None = None
    trace_output_dir: Path | None = None
    max_traces: int = 5000
    pca_components: int = 50


class CoverageObserver:
    def __init__(self, config: CoverageObserverConfig) -> None:
        self.config = config
        self._status = "not-started"
        self._notes: list[str] = []
        self._hook_handle: Any | None = None
        self._latest_trace: np.ndarray | None = None
        self._traces: list[np.ndarray] = []
        self._trace_dim = 0
        self._hook_calls = 0
        self._hook_seconds = 0.0
        self._covered_sections: set[tuple[int, int]] = set()
        self._lsa_scores: list[float] = []
        self._profile: dict[str, Any] | None = None
        self._torch: Any | None = None
        self._joblib: Any | None = None
        self._profile_shape_warning_emitted = False

    def on_run_start(self, scenario: ScenarioSpec, context: dict[str, Any]) -> None:
        del scenario
        # Per-tick scoring is one sample through scaler/PCA/KDE. A multi-threaded
        # BLAS gains nothing there, and its spinning worker threads took CPU from
        # the agent and the CARLA servers (a 2-server batch ran 49 s -> 37.5 s
        # with BLAS at 1 thread).
        try:
            from threadpoolctl import threadpool_limits

            self._blas_limits = threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            pass
        self._status = "initializing"
        self._notes = []
        self._latest_trace = None
        self._traces = []
        self._covered_sections = set()
        self._lsa_scores = []
        self._hook_calls = 0
        self._hook_seconds = 0.0
        self._trace_dim = 0
        self._profile = None
        self._joblib = None
        self._profile_shape_warning_emitted = False

        try:
            import torch  # type: ignore
        except ModuleNotFoundError:
            self._status = "torch-unavailable"
            self._notes.append("PyTorch is not installed in the harness environment.")
            context["coverage_observer"] = self
            return

        self._torch = torch
        agent = context.get("agent")
        model_info = self._inspect_agent_model(agent)
        if model_info is not None:
            context["coverage_model_info"] = model_info

        hook_info = self._register_agent_hook(agent)
        if hook_info is not None:
            self._hook_handle = hook_info["handle"]
            context["coverage_target_layer"] = hook_info.get("layer_name")
            if "model_summary" in hook_info:
                context["coverage_model_info"] = hook_info["model_summary"]
            self._notes.append(
                f"Registered activation hook via agent bridge on layer '{hook_info.get('layer_name')}'."
            )
            self._status = "collecting"
        else:
            model = self._resolve_instrumentable_model(agent)
            if model is None:
                self._status = "unsupported-agent"
                self._notes.append("Agent does not expose an instrumentable PyTorch model.")
                context["coverage_observer"] = self
                return

            module_name, module = self._resolve_target_module(model)
            if module is None:
                self._status = "missing-layer"
                self._notes.append("Could not resolve a target layer for activation hooks.")
                context["coverage_observer"] = self
                return

            self._hook_handle = module.register_forward_hook(self._make_hook())
            context["coverage_target_layer"] = module_name
            self._notes.append(f"Registered activation hook directly on layer '{module_name}'.")
            self._status = "collecting"

        context["coverage_observer"] = self

        if self.config.profile_path is not None:
            try:
                import joblib  # type: ignore
            except ModuleNotFoundError:
                self._status = "profile-unavailable"
                self._notes.append("joblib is not installed, so the coverage profile could not be loaded.")
                return
            self._joblib = joblib
            try:
                self._profile = joblib.load(self.config.profile_path)
            except (OSError, ValueError, EOFError) as exc:
                # A campaign must still collect traces and report collisions when
                # the nominal profile has not been fitted yet (e.g. frozen-seed
                # ADS validation). KMNC/LSA stay null; the run is not aborted.
                self._profile = None
                self._status = "profile-load-failed"
                self._notes.append(f"Could not load coverage profile {self.config.profile_path}: {exc!r}")
                return
            self._status = "profile-loaded"

    def on_tick(self, tick_index: int, context: dict[str, Any]) -> None:
        del tick_index, context
        if self._latest_trace is None:
            return

        trace = self._latest_trace
        self._latest_trace = None
        self._trace_dim = trace.shape[0]
        if len(self._traces) < self.config.max_traces:
            self._traces.append(trace)

        if self._profile is None:
            return

        profile_trace = self._project_trace_to_profile(trace)
        if profile_trace is None:
            return

        mins = self._profile["mins"]
        maxs = self._profile["maxs"]
        k_sections = int(self._profile["k_sections"])
        self._update_kmnc(profile_trace, mins, maxs, k_sections)
        lsa_score = self._compute_lsa(profile_trace)
        if lsa_score is not None:
            self._lsa_scores.append(float(lsa_score))

    def on_run_end(self, result: RunResult, context: dict[str, Any]) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        trace_path: str | None = None
        if self.config.trace_output_dir is not None and self._traces:
            trace_path = str(self._write_trace_dump(result, self.config.trace_output_dir))

        total_sections = None
        kmnc = None
        if self._profile is not None:
            total_sections = int(self._profile["mins"].shape[0]) * int(self._profile["k_sections"])
            kmnc = len(self._covered_sections) / total_sections if total_sections else None

        result.metadata["coverage"] = {
            "status": self._status,
            "notes": list(self._notes),
            "target_layer": self._profile.get("layer_name") if self._profile is not None else None,
            "resolved_target_layer": context.get("coverage_target_layer"),
            "model_info": context.get("coverage_model_info"),
            "hook_calls": self._hook_calls,
            "hook_seconds": self._hook_seconds,
            "trace_count": len(self._traces),
            "trace_dim": self._trace_dim,
            "kmnc": kmnc,
            "kmnc_neuron_coverage": (
                len({neuron_index for neuron_index, _section_index in self._covered_sections}) / int(self._profile["mins"].shape[0])
                if self._profile is not None and int(self._profile["mins"].shape[0])
                else None
            ),
            "covered_sections": len(self._covered_sections) if self._profile is not None else None,
            "total_sections": total_sections,
            "lsa_mean": float(np.mean(self._lsa_scores)) if self._lsa_scores else None,
            "lsa_max": float(np.max(self._lsa_scores)) if self._lsa_scores else None,
            "lsa_p95": float(np.percentile(self._lsa_scores, 95)) if self._lsa_scores else None,
            "profile_path": str(self.config.profile_path) if self.config.profile_path is not None else None,
            "trace_dump_path": trace_path,
        }

    def _make_hook(self) -> Any:
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            start = perf_counter()
            trace = self._flatten_output(output)
            self._latest_trace = trace
            self._hook_calls += 1
            self._hook_seconds += perf_counter() - start

        return _hook

    def _flatten_output(self, output: Any) -> np.ndarray:
        torch = self._torch
        if torch is None:
            raise RuntimeError("PyTorch was expected but is not available.")

        if isinstance(output, torch.Tensor):
            tensor = output.detach().float().cpu()
        elif isinstance(output, (tuple, list)) and output:
            return self._flatten_output(output[0])
        elif isinstance(output, dict) and output:
            first_value = next(iter(output.values()))
            return self._flatten_output(first_value)
        else:
            raise TypeError(f"Unsupported hook output type: {type(output)!r}")

        if tensor.ndim == 0:
            return tensor.reshape(1).numpy()
        if tensor.ndim == 1:
            return tensor.numpy()
        return tensor.reshape(tensor.shape[0], -1)[0].numpy()

    def _inspect_agent_model(self, agent: Any) -> dict[str, Any] | None:
        inspect_fn = getattr(agent, "inspect_loaded_model", None)
        if callable(inspect_fn):
            try:
                return inspect_fn()
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"Model inspection via agent bridge failed: {exc!r}")
        return None

    def _register_agent_hook(self, agent: Any) -> dict[str, Any] | None:
        register_fn = getattr(agent, "register_activation_hook", None)
        if callable(register_fn):
            try:
                return register_fn(self._make_hook(), layer_name=self.config.layer_name)
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"Dynamic hook registration via agent bridge failed: {exc!r}")
        return None

    def _resolve_instrumentable_model(self, agent: Any) -> Any | None:
        if agent is None:
            return None
        candidates = [
            getattr(agent, "torch_model", None),
            getattr(agent, "model", None),
            getattr(agent, "network", None),
            getattr(agent, "net", None),
            getattr(agent, "policy", None),
        ]
        if self._torch is not None:
            candidates.append(agent if isinstance(agent, self._torch.nn.Module) else None)
        for candidate in candidates:
            if candidate is not None and self._torch is not None and isinstance(candidate, self._torch.nn.Module):
                return candidate
        return None

    def _resolve_target_module(self, model: Any) -> tuple[str | None, Any | None]:
        if self.config.layer_name is not None:
            modules = dict(model.named_modules())
            return self.config.layer_name, modules.get(self.config.layer_name)

        named_modules = list(model.named_modules())
        if self._torch is not None:
            for name, module in reversed(named_modules):
                if isinstance(module, self._torch.nn.Linear):
                    return name, module
        for name, module in reversed(named_modules):
            params = list(module.parameters(recurse=False))
            if params:
                return name, module
        return None, None

    def _update_kmnc(self, trace: np.ndarray, mins: np.ndarray, maxs: np.ndarray, k_sections: int) -> None:
        widths = maxs - mins
        valid_mask = widths > 1e-12
        if not np.any(valid_mask):
            return
        within_bounds = (trace >= mins) & (trace <= maxs)
        scaled = np.zeros_like(trace)
        scaled[valid_mask] = (trace[valid_mask] - mins[valid_mask]) / widths[valid_mask]
        section_ids = np.floor(scaled * k_sections).astype(int)
        section_ids = np.clip(section_ids, 0, k_sections - 1)
        for neuron_index, section_index in enumerate(section_ids):
            if valid_mask[neuron_index] and within_bounds[neuron_index]:
                self._covered_sections.add((neuron_index, int(section_index)))

    def _compute_lsa(self, trace: np.ndarray) -> float | None:
        if self._profile is None:
            return None
        scaler = self._profile.get("scaler")
        pca = self._profile.get("pca")
        kde = self._profile.get("kde")
        if scaler is None or pca is None or kde is None:
            return None
        scaled = scaler.transform(trace.reshape(1, -1))
        latent = pca.transform(scaled)
        log_density = kde.score_samples(latent)[0]
        return float(-log_density)

    def _project_trace_to_profile(self, trace: np.ndarray) -> np.ndarray | None:
        if self._profile is None:
            return trace

        raw_trace_dim = int(self._profile.get("raw_trace_dim", trace.shape[0]))
        if trace.shape[0] != raw_trace_dim:
            if not self._profile_shape_warning_emitted:
                self._notes.append(
                    f"Trace dimension {trace.shape[0]} does not match profile raw dimension {raw_trace_dim}; skipping profile scoring."
                )
                self._profile_shape_warning_emitted = True
            return None

        valid_mask = self._profile.get("valid_mask")
        if valid_mask is None:
            return trace
        return trace[np.asarray(valid_mask, dtype=bool)]

    def _write_trace_dump(self, result: RunResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{result.scenario_id}-{result.agent_kind}-{timestamp}-traces.npz"
        np.savez_compressed(out_path, traces=np.stack(self._traces, axis=0))
        return out_path


def build_coverage_profile(
    traces: np.ndarray,
    *,
    k_sections: int,
    pca_components: int,
    layer_name: str,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    if traces.ndim != 2:
        raise ValueError(f"Expected a 2D trace array, got shape {traces.shape!r}")
    if traces.shape[0] < 2:
        raise ValueError("Need at least two traces to build a coverage profile.")

    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.neighbors import KernelDensity  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    valid_mask = np.isfinite(traces).all(axis=0) & ((traces.max(axis=0) - traces.min(axis=0)) > 1e-12)
    filtered_traces = traces[:, valid_mask]
    if filtered_traces.shape[1] == 0:
        raise ValueError("All trace dimensions are constant or non-finite; cannot build a KMNC/LSA profile.")

    mins = filtered_traces.min(axis=0)
    maxs = filtered_traces.max(axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(filtered_traces)
    effective_components = max(1, min(pca_components, scaled.shape[0], scaled.shape[1]))
    pca = PCA(n_components=effective_components)
    latent = pca.fit_transform(scaled)
    resolved_bandwidth = _estimate_kde_bandwidth(latent) if bandwidth is None else float(bandwidth)
    kde = KernelDensity(kernel="gaussian", bandwidth=resolved_bandwidth)
    kde.fit(latent)
    return {
        "layer_name": layer_name,
        "k_sections": int(k_sections),
        "mins": mins,
        "maxs": maxs,
        "valid_mask": valid_mask,
        "raw_trace_dim": int(traces.shape[1]),
        "effective_trace_dim": int(filtered_traces.shape[1]),
        "trace_count": int(traces.shape[0]),
        "bandwidth": float(resolved_bandwidth),
        "scaler": scaler,
        "pca": pca,
        "kde": kde,
        "trace_shape": traces.shape,
    }


def score_traces_against_profile(traces: np.ndarray, profile: dict[str, Any]) -> dict[str, Any]:
    if traces.ndim != 2:
        raise ValueError(f"Expected a 2D trace array, got shape {traces.shape!r}")

    projected_traces = _project_traces_for_profile(traces, profile)
    mins = np.asarray(profile["mins"])
    maxs = np.asarray(profile["maxs"])
    k_sections = int(profile["k_sections"])
    covered_sections, in_range_count, comparable_count = _collect_kmnc_hits(projected_traces, mins, maxs, k_sections)
    lsa_scores = _compute_lsa_scores(projected_traces, profile)
    effective_dim = int(mins.shape[0])
    total_sections = effective_dim * k_sections

    return {
        "layer_name": profile.get("layer_name"),
        "trace_count": int(projected_traces.shape[0]),
        "raw_trace_dim": int(profile.get("raw_trace_dim", traces.shape[1])),
        "effective_trace_dim": effective_dim,
        "k_sections": k_sections,
        "covered_sections": len(covered_sections),
        "total_sections": total_sections,
        "kmnc": len(covered_sections) / total_sections if total_sections else None,
        "kmnc_neuron_coverage": (
            len({neuron_index for neuron_index, _section_index in covered_sections}) / effective_dim if effective_dim else None
        ),
        "out_of_range_fraction": (
            1.0 - (in_range_count / comparable_count) if comparable_count else None
        ),
        "lsa_mean": float(np.mean(lsa_scores)) if lsa_scores.size else None,
        "lsa_max": float(np.max(lsa_scores)) if lsa_scores.size else None,
        "lsa_min": float(np.min(lsa_scores)) if lsa_scores.size else None,
        "lsa_p95": float(np.percentile(lsa_scores, 95)) if lsa_scores.size else None,
        "lsa_scores": lsa_scores.tolist(),
        "bandwidth": float(profile.get("bandwidth")) if profile.get("bandwidth") is not None else None,
    }


def _project_traces_for_profile(traces: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    raw_trace_dim = int(profile.get("raw_trace_dim", traces.shape[1]))
    if traces.shape[1] != raw_trace_dim:
        raise ValueError(
            f"Trace dimension {traces.shape[1]} does not match profile raw dimension {raw_trace_dim}."
        )
    valid_mask = profile.get("valid_mask")
    if valid_mask is None:
        return traces
    return traces[:, np.asarray(valid_mask, dtype=bool)]


def _collect_kmnc_hits(
    traces: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    k_sections: int,
) -> tuple[set[tuple[int, int]], int, int]:
    widths = maxs - mins
    valid_widths = widths > 1e-12
    if not np.any(valid_widths):
        return set(), 0, 0

    within_bounds = (traces >= mins) & (traces <= maxs) & valid_widths[np.newaxis, :]
    scaled = np.zeros_like(traces)
    scaled[:, valid_widths] = (traces[:, valid_widths] - mins[valid_widths]) / widths[valid_widths]
    section_ids = np.floor(scaled * k_sections).astype(int)
    section_ids = np.clip(section_ids, 0, k_sections - 1)

    covered_sections: set[tuple[int, int]] = set()
    row_indices, neuron_indices = np.nonzero(within_bounds)
    for row_index, neuron_index in zip(row_indices, neuron_indices):
        covered_sections.add((int(neuron_index), int(section_ids[row_index, neuron_index])))

    in_range_count = int(np.count_nonzero(within_bounds))
    comparable_count = int(traces.shape[0] * np.count_nonzero(valid_widths))
    return covered_sections, in_range_count, comparable_count


def _compute_lsa_scores(traces: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    scaler = profile.get("scaler")
    pca = profile.get("pca")
    kde = profile.get("kde")
    if scaler is None or pca is None or kde is None:
        return np.asarray([], dtype=float)
    scaled = scaler.transform(traces)
    latent = pca.transform(scaled)
    return -kde.score_samples(latent)


def _estimate_kde_bandwidth(latent: np.ndarray) -> float:
    if latent.ndim != 2 or latent.shape[0] < 2:
        return 1.0
    dimension = max(1, latent.shape[1])
    std = float(np.mean(np.std(latent, axis=0, ddof=1)))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    factor = latent.shape[0] ** (-1.0 / (dimension + 4.0))
    return max(std * factor, 1e-3)