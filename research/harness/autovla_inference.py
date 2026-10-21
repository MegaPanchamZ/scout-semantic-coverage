"""In-memory inference preserving released adaptive decoding by default."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


ADAPTIVE_SYSTEM_PROMPT = 'You are an Advanced Driver Assistance and Full Self-Driving System. You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. Your task is to predict the optimal driving action for the next five seconds.\n\nFirst, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\nIf necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\nStructure your reasoning as follows:\n1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\nPresent the final action clearly after your reasoning steps.'


DIRECT_ANSWER_PREFIX = (
    "<think>\nThis is a straightforward scenario, and a direct decision can be made.\n"
    "</think>\n<answer>\nThe final output action is: "
)


def validate_trajectory(poses: Any) -> np.ndarray:
    if hasattr(poses, "detach"):
        poses = poses.detach().float().cpu().numpy()
    result = np.asarray(poses, dtype=np.float32)
    if result.shape != (10, 3) or not np.isfinite(result).all():
        raise ValueError("AutoVLA must produce ten finite (x, y, heading) poses")
    return result.copy()


class AutoVlaInference:
    def __init__(self, model: Any, decoding_mode: str = "adaptive") -> None:
        if decoding_mode not in {"adaptive", "direct"}:
            raise ValueError("AutoVLA decoding mode must be adaptive or direct")
        self.decoding_mode = decoding_mode
        self.model = model
        self.last_metrics: dict[str, Any] = {}
        tokenizer = model.processor.tokenizer
        action_ids = [tokenizer.convert_tokens_to_ids(f"<action_{i}>")
                      for i in range(model.action_tokenizer.n_bins)]
        self.action_start = model.action_start_id
        if action_ids != list(range(self.action_start, self.action_start + len(action_ids))):
            raise ValueError("AutoVLA checkpoint/tokenizer action vocabulary does not match the codebook")
        self.action_end = self.action_start + len(action_ids)
        model.vlm.set_attn_implementation("sdpa")

    def _prompt(self, features: dict[str, Any]) -> Any:
        from qwen_vl_utils import process_vision_info

        model = self.model
        views = [
            ("front_camera", "front view"),
            ("front_left_camera", "front-left view"),
            ("front_right_camera", "front-right view"),
        ]
        content = [{"type": "text", "text": (
            "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, "
            "enabling a comprehensive perception of the surrounding environment."
        )}]
        for ordinal, (name, description) in zip(("first", "second", "third"), views):
            frames = features["images"][name]
            if len(frames) != 4:
                raise ValueError("AutoVLA requires four synchronized frames per camera")
            content.extend([
                {"type": "text", "text": (
                    f"The {ordinal} video presents the {description} of the vehicle, "
                    "comprising four sequential frames sampled at 2 Hz."
                )},
                {"type": "video", "video": list(frames), "fps": 2.0,
                 "min_pixels": model.video_conf["min_pixels"],
                 "max_pixels": model.video_conf["max_pixels"]},
            ])
        velocity = float(np.linalg.norm(features["vehicle_velocity"]))
        acceleration = float(np.linalg.norm(features["vehicle_acceleration"]))
        content.append({"type": "text", "text": (
            f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is "
            f"{acceleration:.3f} m/s². The driving instruction is: {features['driving_command'].lower()}. "
            "Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
        )})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle’s surrounding cameras, "
                "along with the vehicle’s current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )}]},
            {"role": "user", "content": content},
        ]
        if self.decoding_mode == "adaptive":
            messages[0]["content"][0]["text"] = ADAPTIVE_SYSTEM_PROMPT
        images, videos = process_vision_info(messages)
        text = model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        )
        if self.decoding_mode == "direct":
            text += DIRECT_ANSWER_PREFIX
        return model.processor(text=[text], images=images, videos=videos,
                               padding=True, return_tensors="pt")

    def predict(self, features: dict[str, Any]) -> tuple[np.ndarray, str]:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        start_id, end_id = self.action_start, self.action_end

        class ActionTokensOnly(LogitsProcessor):
            def __call__(self, input_ids: Any, scores: Any) -> Any:
                if self_runtime.decoding_mode == "direct" or start_id <= int(input_ids[0, -1]) < end_id:
                    scores[:, :start_id] = -torch.inf
                    scores[:, end_id:] = -torch.inf
                return scores

        self_runtime = self

        class TenActions(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                return ((input_ids >= start_id) & (input_ids < end_id)).sum(dim=1) >= 10

        started = time.perf_counter()
        with torch.inference_mode():
            inputs = self._prompt(features)
            model_inputs = {key: value.to(self.model.device) for key, value in inputs.items()
                            if isinstance(value, torch.Tensor)}
            prepared = time.perf_counter()
            outputs = self.model.vlm.generate(
                **model_inputs, max_new_tokens=10 if self.decoding_mode == "direct" else 384,
                do_sample=self.decoding_mode == "adaptive", use_cache=True,
                **({"temperature": 0.01, "top_k": 0, "top_p": 1.0}
                   if self.decoding_mode == "adaptive" else {}),
                stopping_criteria=StoppingCriteriaList([TenActions()]),
                logits_processor=LogitsProcessorList([ActionTokensOnly()]),
                eos_token_id=None, forced_eos_token_id=None,
            )
            completion = outputs[0, inputs.input_ids.shape[1]:].cpu()
            actions = completion if self.decoding_mode == "direct" else completion[(completion >= start_id) & (completion < end_id)]
            generated = time.perf_counter()
            if len(actions) != 10 or not ((actions >= start_id) & (actions < end_id)).all():
                raise ValueError("AutoVLA did not generate exactly ten valid action tokens")
            poses = validate_trajectory(self.model.action_tokenizer.decode_token_ids_to_trajectory(actions)[0, 1:])
        self.last_metrics = {
            "preprocessing_seconds": prepared - started,
            "generation_seconds": generated - prepared,
            "inference_seconds": time.perf_counter() - started,
            "prompt_tokens": int(inputs.input_ids.shape[1]), "action_tokens": 10,
            "action_ids": actions.tolist(), "attention": "sdpa", "decoding": "sampled-adaptive" if self.decoding_mode == "adaptive" else "greedy-direct",
        }
        return poses, self.model.processor.decode(completion)
