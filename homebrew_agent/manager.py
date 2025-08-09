from homebrew_agent.agent import AgentFactory, AgentInfo
from homebrew_agent.logging_config import setup_logging
from typing import Optional
import time
import asyncio
from dataclasses import dataclass

logger = setup_logging(__name__)


@dataclass
class CachedAgentInfo:
    agent_info: AgentInfo
    created_at: float
    last_accessed: float
    access_count: int = 0


class AgentManager:
    db_url: str
    agent_factory: AgentFactory
    agents: dict[str, CachedAgentInfo]
    max_cache_size: int
    cache_ttl: float  # Time to live in seconds
    cleanup_interval: float  # Cleanup interval in seconds
    _cleanup_task: Optional[asyncio.Task]

    def __init__(
        self,
        db_url: str,
        max_cache_size: int = 100,
        cache_ttl: float = 3600.0,  # 1 hour
        cleanup_interval: float = 300.0,  # 5 minutes
    ):
        self.db_url = db_url
        self.agent_factory = AgentFactory(db_url)
        self.agents = {}
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.cleanup_interval = cleanup_interval
        self._cleanup_task = None

    async def start_cleanup_task(self):
        """Start the periodic cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop_cleanup_task(self):
        """Stop the periodic cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _periodic_cleanup(self):
        """Periodically clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")

    async def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = []

        for agent_id, cached_info in self.agents.items():
            if current_time - cached_info.created_at > self.cache_ttl:
                expired_keys.append(agent_id)

        for key in expired_keys:
            del self.agents[key]
            logger.info(f"Removed expired agent from cache: {key}")

    def _evict_lru_entry(self):
        """Evict the least recently used entry when cache is full"""
        if not self.agents:
            return

        # Find the least recently accessed entry
        lru_key = min(self.agents.keys(), key=lambda k: self.agents[k].last_accessed)
        del self.agents[lru_key]
        logger.info(f"Evicted LRU agent from cache: {lru_key}")

    async def get_or_create_agent_by_id(self, agent_id: str) -> Optional[AgentInfo]:
        current_time = time.time()

        # Check if agent exists in cache and is not expired
        if agent_id in self.agents:
            cached_info = self.agents[agent_id]

            # Check if entry is expired
            if current_time - cached_info.created_at > self.cache_ttl:
                del self.agents[agent_id]
                logger.info(f"Removed expired agent from cache: {agent_id}")
            else:
                # Update access information
                cached_info.last_accessed = current_time
                cached_info.access_count += 1
                logger.debug(f"Cache hit for agent: {agent_id}")
                return cached_info.agent_info

        # Agent not in cache or expired, create new one
        logger.info(f"Creating new agent: {agent_id}")
        agent_info = await self.agent_factory.build_agent_starlette_by_id(agent_id)

        if agent_info:
            # Check cache size limit and evict if necessary
            if len(self.agents) >= self.max_cache_size:
                self._evict_lru_entry()

            # Add to cache
            self.agents[agent_id] = CachedAgentInfo(
                agent_info=agent_info,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
            )
            logger.info(f"Added agent to cache: {agent_id}")

        return agent_info

    async def invalidate_agent(self, agent_id: str):
        """Manually invalidate a specific agent from cache"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Manually invalidated agent from cache: {agent_id}")

    async def clear_cache(self):
        """Clear all cached agents"""
        self.agents.clear()
        logger.info("Cleared all agents from cache")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        current_time = time.time()
        total_entries = len(self.agents)
        expired_entries = sum(
            1
            for cached_info in self.agents.values()
            if current_time - cached_info.created_at > self.cache_ttl
        )

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "max_cache_size": self.max_cache_size,
            "cache_ttl": self.cache_ttl,
            "cleanup_interval": self.cleanup_interval,
        }
