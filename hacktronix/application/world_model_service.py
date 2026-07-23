"""
World Model Service – Dependency Injection Container.

Creates and wires all infrastructure & application components together.
"""

from hacktronix.infrastructure.db.database import DatabaseManager
from hacktronix.infrastructure.db.repositories import SQLiteWorldRepository
from hacktronix.infrastructure.graph_store import NetworkXGraphStore
from hacktronix.infrastructure.vector_store import FAISSVectorStore
from hacktronix.infrastructure.llm.mock_llm import MockLLMProvider
from hacktronix.infrastructure.llm.ollama_client import OllamaLLMProvider
from hacktronix.application.extractor import ObservationExtractor
from hacktronix.application.updater import UpdaterEngine
from hacktronix.application.query_layer import QueryLayer
from hacktronix.application.agent import TextWorldAgent
from hacktronix.environment.text_env import TextAdventureEnv
from hacktronix.environment.video_env import VideoEnvironment


def build_world_model_stack(db_path: str = "data/world_model.db", use_ollama: bool = False):
    """
    Builds the full World Model service stack and returns named components.

    Args:
        db_path: SQLite database file path.
        use_ollama: If True, attempts to connect to Ollama; else uses deterministic MockLLM.

    Returns:
        Dictionary of all constructed service components.
    """
    # Infrastructure
    db_manager = DatabaseManager(db_path)
    repository = SQLiteWorldRepository(db_manager)
    graph_store = NetworkXGraphStore()
    vector_store = FAISSVectorStore()

    # LLM
    if use_ollama:
        llm = OllamaLLMProvider()
    else:
        llm = MockLLMProvider()

    # Application services
    extractor = ObservationExtractor()
    updater = UpdaterEngine(repository, graph_store, vector_store)
    query_layer = QueryLayer(repository, graph_store, vector_store)

    # Environments
    text_env = TextAdventureEnv()
    video_env = VideoEnvironment()

    # Agent
    agent = TextWorldAgent(
        env=text_env,
        llm=llm,
        query_layer=query_layer,
        extractor=extractor,
        updater=updater,
    )

    return {
        "db_manager": db_manager,
        "repository": repository,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "llm": llm,
        "extractor": extractor,
        "updater": updater,
        "query_layer": query_layer,
        "text_env": text_env,
        "video_env": video_env,
        "agent": agent,
    }
