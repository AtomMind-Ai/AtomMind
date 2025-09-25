"""Generators for Chat and Science tasks using ExecutorAgent."""

from typing import List
from agents.executor import ExecutorAgent

class ChatGenerator:
    """
    Generates daily conversational examples using the ExecutorAgent.
    """

    def __init__(self, executor: ExecutorAgent):
        self.executor = executor

    def generate(self, num_candidates: int = 1) -> List[str]:
        """
        Generate daily chat examples.

        Args:
            num_candidates (int): Number of chat examples to generate.

        Returns:
            List[str]: Generated chat examples.
        """
        prompt = (
            "Your task is to generate a dataset for training a chatbot. "
            "Each dataset entry should be a realistic short daily conversation "
            "between two people. Do not include labels like 'User1:' or 'User2:'. "
            "Instead, just alternate lines of dialogue. "
            "Limit each example from 5 to 10 sentences total."
        )

        return self.executor.generate_candidates(
            prompt=prompt,
            num_candidates=num_candidates,
            system_prompt="You are a professional prompt engineer.",
            max_tokens=2000,
            temperature=0.7
        )

class ScienceGenerator:
    """
    Generates scientific tasks/problems using the ExecutorAgent.
    """

    def __init__(self, executor: ExecutorAgent):
        self.executor = executor

    def generate(self, num_candidates: int = 3) -> List[str]:
        """
        Generate scientific tasks/problems.

        Args:
            num_candidates (int): Number of tasks to generate.

        Returns:
            List[str]: Generated science tasks/problems.
        """
        prompt = (
            "Generate a small, short scientific task/problem in math, physics, "
            "chemistry, or biology."
        )
        return self.executor.generate_candidates(
            prompt=prompt,
            num_candidates=num_candidates,
            system_prompt="You are a professional scientific assistant.",
            max_tokens=2000,
            temperature=0.7
        )
