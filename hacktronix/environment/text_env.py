"""
Text Adventure Environment Simulator.

Provides a rich, multi-room text adventure world with rooms, objects,
doors, dynamic events, and a structured observation API.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Room:
    """Represents a room in the text adventure world."""
    id: str
    name: str
    description: str
    exits: Dict[str, str] = field(default_factory=dict)   # direction -> room_id
    objects: List[str] = field(default_factory=list)       # object IDs present
    events: List[str] = field(default_factory=list)        # dynamic event log


@dataclass
class WorldObject:
    """Represents an interactive object in the world."""
    id: str
    name: str
    room_id: Optional[str]
    states: Dict[str, str] = field(default_factory=dict)
    takeable: bool = True
    description: str = ""


class TextAdventureEnv:
    """
    Simulates a multi-room text adventure world. Acts as Track 1 environment.
    Maintains agent position, inventory, and full room-object graph.
    """

    VALID_ACTIONS = ["explore", "examine", "take", "drop", "use", "open", "unlock", "go"]

    def __init__(self) -> None:
        self.rooms: Dict[str, Room] = {}
        self.objects: Dict[str, WorldObject] = {}
        self.agent_room_id: str = ""
        self.inventory: List[str] = []
        self.step_count: int = 0
        self._build_default_world()

    def _build_default_world(self) -> None:
        """Builds a default 6-room adventure world for demo purposes."""
        # Rooms
        self.rooms = {
            "entrance_hall": Room(
                id="entrance_hall",
                name="Entrance Hall",
                description="A grand stone entrance hall. Torches flicker on the walls. There are heavy wooden doors to the north and east.",
                exits={"north": "library", "east": "armory"},
                objects=["torch_1", "notice_board"],
            ),
            "library": Room(
                id="library",
                name="Ancient Library",
                description="Floor-to-ceiling bookshelves fill this dusty chamber. A locked iron chest sits beneath a reading table.",
                exits={"south": "entrance_hall", "west": "secret_passage"},
                objects=["iron_chest", "brass_key"],
            ),
            "armory": Room(
                id="armory",
                name="Weapons Armory",
                description="Racks of rusting weapons line the walls. A locked door leads north to the dungeon.",
                exits={"west": "entrance_hall", "north": "dungeon"},
                objects=["iron_sword", "shield_1"],
            ),
            "secret_passage": Room(
                id="secret_passage",
                name="Secret Passage",
                description="A narrow, hidden corridor. It smells of mold. There is a glowing orb on a pedestal.",
                exits={"east": "library", "north": "throne_room"},
                objects=["glowing_orb"],
            ),
            "dungeon": Room(
                id="dungeon",
                name="Dungeon Cell",
                description="A damp stone cell. A prisoner sits chained to the wall. There is a rusty key on the floor.",
                exits={"south": "armory"},
                objects=["rusty_key", "prisoner_note"],
            ),
            "throne_room": Room(
                id="throne_room",
                name="Throne Room",
                description="A vast hall with a golden throne. The exit gem, the GOAL ARTIFACT, rests on the throne.",
                exits={"south": "secret_passage"},
                objects=["exit_gem", "throne"],
            ),
        }

        # Objects
        self.objects = {
            "torch_1":       WorldObject("torch_1", "Wall Torch", "entrance_hall", {"lit": "true"}, takeable=False, description="A burning torch mounted on the wall."),
            "notice_board":  WorldObject("notice_board", "Notice Board", "entrance_hall", {}, takeable=False, description="A board with a message: 'Find the Exit Gem to win.'"),
            "iron_chest":    WorldObject("iron_chest", "Iron Chest", "library", {"locked": "true", "open": "false"}, takeable=False, description="A heavy iron chest, locked with a brass lock."),
            "brass_key":     WorldObject("brass_key", "Brass Key", "library", {"material": "brass"}, description="A small brass key. It might open something."),
            "iron_sword":    WorldObject("iron_sword", "Iron Sword", "armory", {"sharpness": "medium"}, description="A standard iron sword."),
            "shield_1":      WorldObject("shield_1", "Wooden Shield", "armory", {"durability": "high"}, description="A battered wooden shield."),
            "glowing_orb":   WorldObject("glowing_orb", "Glowing Orb", "secret_passage", {"glow": "true"}, description="A sphere of pulsing light."),
            "rusty_key":     WorldObject("rusty_key", "Rusty Key", "dungeon", {"material": "iron", "condition": "rusty"}, description="A rusty iron key. Might open an old lock."),
            "prisoner_note": WorldObject("prisoner_note", "Prisoner's Note", "dungeon", {}, description="A note reading: 'The throne holds the answer.'"),
            "exit_gem":      WorldObject("exit_gem", "EXIT GEM (GOAL)", "throne_room", {"magical": "true"}, description="The goal artifact. Glowing with divine energy."),
            "throne":        WorldObject("throne", "Golden Throne", "throne_room", {"material": "gold"}, takeable=False, description="An ornate golden throne."),
        }

        self.agent_room_id = "entrance_hall"

    def observe(self) -> Dict[str, Any]:
        """Returns a structured observation of the current room and its contents."""
        room = self.rooms.get(self.agent_room_id)
        if not room:
            return {"error": f"No room with ID: {self.agent_room_id}"}

        objects_here = []
        for obj_id in room.objects:
            obj = self.objects.get(obj_id)
            if obj:
                objects_here.append({
                    "id": obj.id,
                    "name": obj.name,
                    "description": obj.description,
                    "states": obj.states,
                    "takeable": obj.takeable,
                })

        inv_objects = []
        for obj_id in self.inventory:
            obj = self.objects.get(obj_id)
            if obj:
                inv_objects.append({"id": obj.id, "name": obj.name})

        return {
            "step": self.step_count,
            "current_room": {
                "id": room.id,
                "name": room.name,
                "description": room.description,
                "exits": room.exits,
            },
            "objects_in_room": objects_here,
            "inventory": inv_objects,
            "events": room.events[-3:] if room.events else [],
        }

    def step(self, action: str, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes an action in the environment and returns the resulting observation.
        """
        self.step_count += 1
        action = action.lower().strip()
        result_msg = ""

        if action == "go" and target:
            result_msg = self._do_go(target)
        elif action == "take" and target:
            result_msg = self._do_take(target)
        elif action == "drop" and target:
            result_msg = self._do_drop(target)
        elif action == "examine" and target:
            result_msg = self._do_examine(target)
        elif action == "open" and target:
            result_msg = self._do_open(target)
        elif action == "use" and target:
            result_msg = self._do_use(target)
        elif action == "explore":
            result_msg = "You look around carefully."
        else:
            result_msg = f"Unknown action: '{action}'. Valid: {', '.join(self.VALID_ACTIONS)}"

        observation = self.observe()
        observation["action_result"] = result_msg
        return observation

    def _do_go(self, direction: str) -> str:
        room = self.rooms[self.agent_room_id]
        direction = direction.lower()
        if direction in room.exits:
            self.agent_room_id = room.exits[direction]
            new_room = self.rooms[self.agent_room_id]
            return f"You move {direction}. You are now in: {new_room.name}."
        return f"There is no exit to the {direction}."

    def _do_take(self, target_name: str) -> str:
        room = self.rooms[self.agent_room_id]
        for obj_id in room.objects:
            obj = self.objects.get(obj_id)
            if obj and target_name.lower() in obj.name.lower():
                if not obj.takeable:
                    return f"{obj.name} cannot be taken."
                room.objects.remove(obj_id)
                self.inventory.append(obj_id)
                obj.room_id = None
                return f"You pick up the {obj.name}."
        return f"No takeable object named '{target_name}' found here."

    def _do_drop(self, target_name: str) -> str:
        for obj_id in self.inventory:
            obj = self.objects.get(obj_id)
            if obj and target_name.lower() in obj.name.lower():
                self.inventory.remove(obj_id)
                self.rooms[self.agent_room_id].objects.append(obj_id)
                obj.room_id = self.agent_room_id
                return f"You drop the {obj.name}."
        return f"No item named '{target_name}' in inventory."

    def _do_examine(self, target_name: str) -> str:
        room = self.rooms[self.agent_room_id]
        all_ids = room.objects + self.inventory
        for obj_id in all_ids:
            obj = self.objects.get(obj_id)
            if obj and target_name.lower() in obj.name.lower():
                states_str = ", ".join([f"{k}={v}" for k, v in obj.states.items()])
                return f"[{obj.name}] {obj.description} States: ({states_str})"
        return f"Nothing named '{target_name}' nearby."

    def _do_open(self, target_name: str) -> str:
        room = self.rooms[self.agent_room_id]
        for obj_id in room.objects:
            obj = self.objects.get(obj_id)
            if obj and target_name.lower() in obj.name.lower():
                if obj.states.get("locked") == "true":
                    has_key = any("key" in self.objects[k].name.lower() for k in self.inventory if k in self.objects)
                    if has_key:
                        obj.states["locked"] = "false"
                        obj.states["open"] = "true"
                        room.events.append(f"Opened {obj.name} with a key.")
                        return f"You use a key to unlock and open {obj.name}!"
                    return f"{obj.name} is locked. You need a key."
                obj.states["open"] = "true"
                return f"You open {obj.name}."
        return f"No object named '{target_name}' to open."

    def _do_use(self, target_name: str) -> str:
        for obj_id in self.inventory:
            obj = self.objects.get(obj_id)
            if obj and target_name.lower() in obj.name.lower():
                return f"You use the {obj.name}. It hums with energy."
        return f"No item named '{target_name}' in inventory to use."

    def is_goal_achieved(self) -> bool:
        """Returns True when the agent has picked up the Exit Gem."""
        return "exit_gem" in self.inventory
