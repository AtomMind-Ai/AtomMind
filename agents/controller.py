"""Controller agent for evaluating training progress and suggesting adjustments."""

from typing import Dict, Any
from agents.executor import ExecutorAgent


class ControllerAgent:
    """
    Uses OpenRouter (through ExecutorAgent) to evaluate training logs
    and provide fine-tuning feedback without restricting creativity.
    """

    def __init__(self, executor: ExecutorAgent):
        self.executor = executor

    def control_stage(self, stage: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper so training loop can call `control_stage` directly.
        """
        logs = f"[Stage: {stage}] Context: {context}"
        result = self.evaluate(logs)

        # Merge adjustments into a flat dict so main.py can use them directly
        feedback = {
            "feedback": result.get("feedback", ""),
            **result.get("adjustments", {})
        }
        return feedback

    def evaluate(self, logs: str) -> Dict[str, Any]:
        """
        Evaluate training logs and suggest improvements.

        Args:
            logs (str): Recent training logs or metrics.

        Returns:
            Dict[str, Any]: Structured feedback and suggested adjustments.
        """
        prompt = (
            "You are the CHECKER agent. Your job is to evaluate the model’s "
            "training progress and provide suggestions. "
            "Goals:\n"
            "- Track performance (loss, stability, consistency).\n"
            "- Suggest adjustments (learning rate, gradient clipping, epochs).\n"
            "- Ensure improvements in fluency, reasoning, and creativity.\n"
            "- DO NOT restrict creativity, logic, or thinking.\n\n"
            "Return structured JSON with the following format:\n"
            "{\n"
            '  "feedback": "Short summary of evaluation",\n'
            '  "adjustments": {\n'
            '    "lr": float (optional),\n'
            '    "grad_clip": float (optional),\n'
            '    "epochs": int (optional),\n'
            '    "stop_training": bool (optional),\n'
            '    "freeze_modules": [list of str] (optional),\n'
            '    "unfreeze_modules": [list of str] (optional)\n'
            "  }\n"
            "}\n\n"
            f"Training Logs:\n{logs}"
        )

        results = self.executor.generate_candidates(
            prompt=prompt,
            num_candidates=1,
            system_prompt="You are a strict evaluator and performance checker.",
            max_tokens=800,
            temperature=0.3,
        )

        # Try to parse JSON-like response
        try:
            parsed = eval(results[0])
            if isinstance(parsed, dict):
                return parsed
            return {"feedback": str(results[0]), "adjustments": {}}
        except Exception:
            return {"feedback": results[0], "adjustments": {}}
