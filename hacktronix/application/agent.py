"""
Autonomous Zero-History Text Agent.

The agent receives only its current Objective and a relevant World Slice.
It uses the LLM to reason and decide the next action, then executes it
in the Text Adventure Environment and feeds the resulting observation back
into the World Model. No conversation history is maintained.
"""

import time
from typing import Any, Dict, List, Optional

from hacktronix.domain.interfaces import ILLMProvider
from hacktronix.application.query_layer import QueryLayer
from hacktronix.application.extractor import ObservationExtractor
from hacktronix.application.updater import UpdaterEngine
from hacktronix.environment.text_env import TextAdventureEnv


class TextWorldAgent:
    """
    Zero-history autonomous agent for text adventure exploration.

    Core Loop:
    1. Query the World Model for a relevant slice (Objective + Slice).
    2. Send (Objective + Slice) to the LLM.
    3. Execute the LLM's chosen action in the environment.
    4. Extract structured Observation from result.
    5. Update World Model with new Observation.
    6. Repeat until goal achieved or max steps exceeded.
    """

    def __init__(
        self,
        env: TextAdventureEnv,
        llm: ILLMProvider,
        query_layer: QueryLayer,
        extractor: ObservationExtractor,
        updater: UpdaterEngine,
        max_steps: int = 20,
    ) -> None:
        self.env = env
        self.llm = llm
        self.query_layer = query_layer
        self.extractor = extractor
        self.updater = updater
        self.max_steps = max_steps
        self.step_count = 0
        self.reasoning_log: List[Dict[str, Any]] = []

    def run(self, objective: str) -> Dict[str, Any]:
        """
        Execute the full agent reasoning loop until goal or max steps.

        Returns:
            Summary dict with success flag, step count, and full reasoning log.
        """
        self.step_count = 0
        self.reasoning_log = []

        # Seed World Model with initial observation
        initial_obs_raw = self.env.observe()
        initial_obs = self.extractor.extract_from_text_obs(initial_obs_raw)
        self.updater.process_observation(initial_obs)

        while self.step_count < self.max_steps:
            self.step_count += 1

            # Check goal
            if self.env.is_goal_achieved():
                self.reasoning_log.append({
                    "step": self.step_count,
                    "status": "GOAL ACHIEVED",
                    "reasoning": "The Exit Gem has been collected!",
                })
                break

            # Get World Slice
            current_room_id = self.env.agent_room_id
            world_slice = self.query_layer.retrieve_slice(
                objective=objective,
                current_room_id=current_room_id,
            )
            world_slice_text = world_slice.format_as_text_slice()

            # LLM decision
            llm_response = self.llm.generate_action(objective, world_slice_text)
            action_verb = llm_response.get("action", "explore")
            target = llm_response.get("target", "")
            reasoning = llm_response.get("reasoning", "")

            # Execute in environment
            env_result = self.env.step(action_verb, target)
            action_result_msg = env_result.get("action_result", "")

            # Extract & update World Model
            new_obs = self.extractor.extract_from_text_obs(env_result)
            update_summary = self.updater.process_observation(new_obs)

            # Log step
            step_log = {
                "step": self.step_count,
                "room": env_result.get("current_room", {}).get("name", ""),
                "reasoning": reasoning,
                "action": f"{action_verb} {target}".strip(),
                "result": action_result_msg,
                "entities_updated": len(update_summary.get("updated", [])),
                "entities_added": len(update_summary.get("added", [])),
                "world_slice_preview": world_slice_text[:300],
            }
            self.reasoning_log.append(step_log)

        return {
            "success": self.env.is_goal_achieved(),
            "steps_taken": self.step_count,
            "max_steps": self.max_steps,
            "objective": objective,
            "reasoning_log": self.reasoning_log,
        }

    def step_once(self, objective: str) -> Dict[str, Any]:
        """
        Execute a single agent step and return the step log entry.
        Useful for interactive streaming demos in the dashboard.
        """
        self.step_count += 1

        current_room_id = self.env.agent_room_id
        world_slice = self.query_layer.retrieve_slice(
            objective=objective,
            current_room_id=current_room_id,
        )
        world_slice_text = world_slice.format_as_text_slice()
        llm_response = self.llm.generate_action(objective, world_slice_text)

        action_verb = llm_response.get("action", "explore")
        target = llm_response.get("target", "")
        env_result = self.env.step(action_verb, target)

        new_obs = self.extractor.extract_from_text_obs(env_result)
        self.updater.process_observation(new_obs)

        return {
            "step": self.step_count,
            "room": env_result.get("current_room", {}).get("name", ""),
            "reasoning": llm_response.get("reasoning", ""),
            "action": f"{action_verb} {target}".strip(),
            "result": env_result.get("action_result", ""),
            "goal_achieved": self.env.is_goal_achieved(),
            "world_slice": world_slice_text,
        }
