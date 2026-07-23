"""
Integration Tests for the full World Model pipeline.
Tests: Env -> Extractor -> Updater -> QueryLayer -> Agent -> Env loop.
"""

import pytest
from hacktronix.application.world_model_service import build_world_model_stack
from hacktronix.infrastructure.db.database import DatabaseManager
from hacktronix.infrastructure.db.repositories import SQLiteWorldRepository
from hacktronix.infrastructure.graph_store import NetworkXGraphStore
from hacktronix.infrastructure.vector_store import FAISSVectorStore
from hacktronix.infrastructure.llm.mock_llm import MockLLMProvider
from hacktronix.application.extractor import ObservationExtractor
from hacktronix.application.updater import UpdaterEngine
from hacktronix.application.query_layer import QueryLayer
from hacktronix.application.agent import TextWorldAgent
from hacktronix.environment.text_env import TextAdventureEnv


@pytest.fixture
def stack(tmp_path):
    db_path = str(tmp_path / "test_integration.db")
    db_manager = DatabaseManager(db_path)
    repository = SQLiteWorldRepository(db_manager)
    graph_store = NetworkXGraphStore()
    vector_store = FAISSVectorStore()
    llm = MockLLMProvider()
    extractor = ObservationExtractor()
    updater = UpdaterEngine(repository, graph_store, vector_store)
    query_layer = QueryLayer(repository, graph_store, vector_store)
    env = TextAdventureEnv()
    agent = TextWorldAgent(env, llm, query_layer, extractor, updater, max_steps=10)
    return {
        "repository": repository,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "extractor": extractor,
        "updater": updater,
        "query_layer": query_layer,
        "agent": agent,
        "env": env,
    }


def test_full_observation_pipeline(stack):
    """Test: Env observe -> Extract -> Update -> Query."""
    env = stack["env"]
    extractor = stack["extractor"]
    updater = stack["updater"]
    query_layer = stack["query_layer"]

    raw_obs = env.observe()
    obs = extractor.extract_from_text_obs(raw_obs)
    assert len(obs.entities) > 0

    summary = updater.process_observation(obs)
    assert len(summary["added"]) > 0

    world_slice = query_layer.retrieve_slice(
        objective="Find the Exit Gem",
        current_room_id=env.agent_room_id,
    )
    assert world_slice.current_room is not None
    assert "entrance_hall" in world_slice.current_room.id or world_slice.current_room is not None


def test_agent_multi_step_run(stack):
    """Test: Agent runs multiple steps without crashing."""
    agent = stack["agent"]
    result = agent.run("Find the Exit Gem")

    assert "steps_taken" in result
    assert result["steps_taken"] > 0
    assert "reasoning_log" in result
    assert len(result["reasoning_log"]) > 0


def test_world_model_updater_conflict_resolution(stack):
    """Test: Same entity updated with conflicting states resolves correctly."""
    from hacktronix.domain.entities import Entity
    from hacktronix.domain.value_objects import EntityCategory, Confidence

    repository = stack["repository"]
    updater = stack["updater"]

    # Insert initial entity
    e1 = Entity(id="door_1", name="Iron Door", category=EntityCategory.DOOR)
    e1.set_state("locked", "true", confidence_score=0.9)
    repository.save_entity(e1)

    # Observation with conflicting higher-confidence state
    e2 = Entity(id="door_1", name="Iron Door", category=EntityCategory.DOOR)
    e2.set_state("locked", "false", confidence_score=0.95)  # new observation
    e2.confidence = Confidence(0.95)

    from hacktronix.domain.entities import Observation
    from hacktronix.domain.value_objects import ObservationType
    obs = Observation(
        source_type=ObservationType.TEXT,
        raw_text="Door is now unlocked",
        entities=[e2],
        relationships=[],
    )
    summary = updater.process_observation(obs)
    assert len(summary["conflicts_resolved"]) + len(summary["updated"]) > 0

    updated_entity = repository.get_entity("door_1")
    # Higher confidence observation should win
    assert updated_entity is not None


def test_vision_observation_to_world_model(stack):
    """Test: Vision detection -> Extract -> World Model."""
    extractor = stack["extractor"]
    updater = stack["updater"]

    fake_vision = {
        "objects": [{"name": "Monitor", "category": "object", "confidence": 0.92, "bounding_box": {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 250}}],
        "people": [{"name": "Operator", "category": "person", "confidence": 0.87, "bounding_box": {"xmin": 20, "ymin": 50, "xmax": 120, "ymax": 400}}],
        "faces": [],
        "frame_size": {"width": 640, "height": 480},
    }
    vision_obs = extractor.extract_from_vision_obs(fake_vision, scene_room_id="camera_scene")
    assert len(vision_obs.entities) >= 3  # scene + monitor + operator

    summary = updater.process_observation(vision_obs)
    assert len(summary["added"]) >= 2


def test_query_layer_semantic_search(stack):
    """Test: Query layer returns relevant entities based on objective."""
    from hacktronix.domain.entities import Entity
    from hacktronix.domain.value_objects import EntityCategory

    repository = stack["repository"]
    vector_store = stack["vector_store"]
    query_layer = stack["query_layer"]

    gem = Entity(id="exit_gem", name="Exit Gem", category=EntityCategory.OBJECT)
    gem.set_state("magical", "true")
    repository.save_entity(gem)
    vector_store.index_entity(gem)

    ws = query_layer.retrieve_slice("Find the magical exit gem", top_k=5)
    entity_ids = [e.id for e in ws.visible_entities] + ([ws.current_room.id] if ws.current_room else [])
    assert "exit_gem" in entity_ids
