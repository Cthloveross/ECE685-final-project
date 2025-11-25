from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CONFIG


@dataclass
class ResidualCapture:
    layer_index: int
    last_residual: Optional[torch.Tensor] = None

    def hook(self, module, inputs, output):  # type: ignore[override]
        # Gemma blocks return (hidden_states, optional_cache)
        # Extract the actual tensor from the tuple
        if isinstance(output, tuple):
            output = output[0]
        # output shape: (batch, seq, hidden)
        self.last_residual = output.detach()


class GemmaInterface:
    """Thin wrapper that exposes activation capture and steering hooks."""

    def __init__(self, model_name: str | None = None):
        model_id = model_name or CONFIG.model.gemma_model_name
        dtype = getattr(torch, CONFIG.model.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.capture = ResidualCapture(CONFIG.model.hook_layer)
        self._activation_handle = None
        self._steering_handle = None

    def __enter__(self):
        self.register_capture_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def register_capture_hook(self):
        if self._activation_handle is None:
            block = self.model.model.layers[self.capture.layer_index]
            self._activation_handle = block.register_forward_hook(self.capture.hook)

    def register_steering_hook(self, steering_vector: torch.Tensor, strength: float):
        def steering_fn(module, inputs, output):
            # Handle tuple output from Gemma blocks
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            shifted = hidden_states.clone()
            steer = steering_vector.to(shifted.device) * strength
            shifted[:, -1, :] += steer
            
            # Return in the same format as received
            return (shifted, output[1]) if is_tuple else shifted

        block = self.model.model.layers[self.capture.layer_index]
        self._steering_handle = block.register_forward_hook(steering_fn)

    def remove_hooks(self):
        for handle in (self._activation_handle, self._steering_handle):
            if handle is not None:
                handle.remove()
        self._activation_handle = None
        self._steering_handle = None

    @contextmanager
    def capture_residual(self, steering_vector: Optional[torch.Tensor] = None, strength: float = 0.0):
        self.register_capture_hook()
        if steering_vector is not None and strength != 0.0:
            self.register_steering_hook(steering_vector, strength)
        try:
            yield self.capture
        finally:
            self.remove_hooks()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        steering_vector: Optional[torch.Tensor] = None,
        strength: float = 0.0,
        **gen_kwargs,
    ) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.capture_residual(steering_vector, strength):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **gen_kwargs,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        residual = None
        if self.capture.last_residual is not None:
            residual = self.capture.last_residual[:, -1, :].detach().cpu()
        return {"text": text, "residual": residual}
