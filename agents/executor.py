"""ExecutorAgent module for generating candidates via OpenRouter API."""

import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

class ExecutorAgent:
    """
    ExecutorAgent interacts with the OpenRouter API to generate candidate responses
    for scientific or conversational prompts.
    """

    def __init__(self, api_key: str = None) -> None:
        """Initialize the ExecutorAgent with API key and base URL."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 3,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Generate candidate outputs from the OpenRouter API.

        Args:
            prompt (str): User/system prompt to send to the model.
            num_candidates (int): Number of candidate responses to return.
            system_prompt (str): Role description for the assistant.
            max_tokens (int): Maximum number of tokens per response.
            temperature (float): Sampling temperature for randomness.

        Returns:
            List[str]: Generated candidate responses.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        params = {
            "model": "x-ai/grok-4-fast:free",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": num_candidates
        }

        response = requests.post(self.base_url, json=params, headers=headers, timeout=15)
        response.raise_for_status()

        try:
            choices = response.json().get("choices", [])
            return [choice["message"]["content"] for choice in choices[:num_candidates]]
        except (KeyError, ValueError) as e:
            raise RuntimeError("Unexpected API response format") from e
