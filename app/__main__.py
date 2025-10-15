import logging
import sys
import os
import click
import uvicorn

from dotenv import load_dotenv
from app.multi_agent_server import MultiAgentApplication
from app.agent_handler import AgentFactory


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=8080)
def main(host, port):
    """Starts the Agent server."""
    try:
        if not os.getenv("DB_URL"):
            raise ValueError("DB_URL environment variable not set.")
        db_url = os.getenv("DB_URL")
        agent_factory = AgentFactory(db_url)
        server = MultiAgentApplication(agent_factory)
        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
