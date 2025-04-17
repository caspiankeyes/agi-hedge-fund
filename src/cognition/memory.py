"""
MemoryShell - Temporal Memory Architecture for Recursive Agents

This module implements the memory shell architecture that enables agents to
maintain persistent memory with configurable decay properties. The memory
shell acts as a cognitive substrate that provides:

- Short-term working memory
- Medium-term episodic memory with decay
- Long-term semantic memory with compression
- Temporal relationship tracking
- Experience-based learning

Internal Note: The memory shell simulates the MEMTRACE and ECHO-LOOP interpretability
shells for modeling memory decay and feedback loops in agent cognition.
"""

import datetime
import math
import uuid
import heapq
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict, deque

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Base memory unit with attribution and decay properties."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = Field(...)
    memory_type: str = Field(...)  # "episodic", "semantic", "working"
    creation_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_access_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    access_count: int = Field(default=1)
    salience: float = Field(default=1.0)  # Initial salience (0-1)
    decay_rate: float = Field(default=0.1)  # Default decay rate per time unit
    associations: Dict[str, float] = Field(default_factory=dict)  # Associated memory IDs with strengths
    source: Optional[str] = Field(default=None)  # Source of the memory (e.g., "observation", "reflection")
    tags: List[str] = Field(default_factory=list)  # Semantic tags for the memory
    
    def update_access(self) -> None:
        """Update access time and count."""
        self.last_access_time = datetime.datetime.now()
        self.access_count += 1
    
    def calculate_current_salience(self) -> float:
        """Calculate current salience based on decay model."""
        # Time since creation in hours
        hours_since_creation = (datetime.datetime.now() - self.creation_time).total_seconds() / 3600
        
        # Apply decay model: exponential decay with access-based reinforcement
        base_decay = math.exp(-self.decay_rate * hours_since_creation)
        access_factor = math.log1p(self.access_count) / 10  # Logarithmic access bonus
        
        # Calculate current salience (capped at 1.0)
        current_salience = min(1.0, self.salience * base_decay * (1 + access_factor))
        
        return current_salience
    
    def add_association(self, memory_id: str, strength: float = 0.5) -> None:
        """
        Add association to another memory.
        
        Args:
            memory_id: ID of memory to associate with
            strength: Association strength (0-1)
        """
        self.associations[memory_id] = strength
    
    def add_tag(self, tag: str) -> None:
        """
        Add semantic tag to memory.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "creation_time": self.creation_time.isoformat(),
            "last_access_time": self.last_access_time.isoformat(),
            "access_count": self.access_count,
            "salience": self.salience,
            "current_salience": self.calculate_current_salience(),
            "decay_rate": self.decay_rate,
            "associations": self.associations,
            "source": self.source,
            "tags": self.tags,
        }


class EpisodicMemory(Memory):
    """Episodic memory representing specific experiences."""
    
    sequence_position: Optional[int] = Field(default=None)  # Position in temporal sequence
    emotional_valence: float = Field(default=0.0)  # Emotional charge (-1 to 1)
    outcome: Optional[str] = Field(default=None)  # Outcome of the experience
    
    def __init__(self, **data):
        data["memory_type"] = "episodic"
        super().__init__(**data)


class SemanticMemory(Memory):
    """Semantic memory representing conceptual knowledge."""
    
    certainty: float = Field(default=0.7)  # Certainty level (0-1)
    contradiction_ids: List[str] = Field(default_factory=list)  # IDs of contradicting memories
    supporting_evidence: List[str] = Field(default_factory=list)  # IDs of supporting memories
    
    def __init__(self, **data):
        data["memory_type"] = "semantic"
        # Semantic memories decay more slowly
        data.setdefault("decay_rate", 0.05)
        super().__init__(**data)
    
    def add_evidence(self, memory_id: str, is_supporting: bool = True) -> None:
        """
        Add supporting or contradicting evidence.
        
        Args:
            memory_id: Memory ID for evidence
            is_supporting: Whether evidence is supporting (True) or contradicting (False)
        """
        if is_supporting:
            if memory_id not in self.supporting_evidence:
                self.supporting_evidence.append(memory_id)
        else:
            if memory_id not in self.contradiction_ids:
                self.contradiction_ids.append(memory_id)
    
    def update_certainty(self, evidence_ratio: float) -> None:
        """
        Update certainty based on supporting/contradicting evidence ratio.
        
        Args:
            evidence_ratio: Ratio of supporting to total evidence (0-1)
        """
        # Blend current certainty with evidence ratio
        self.certainty = 0.7 * self.certainty + 0.3 * evidence_ratio


class WorkingMemory(Memory):
    """Working memory representing active thinking and temporary storage."""
    
    expiration_time: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now() + datetime.timedelta(hours=1))
    priority: int = Field(default=1)  # Priority level (higher = more important)
    
    def __init__(self, **data):
        data["memory_type"] = "working"
        # Working memories decay rapidly
        data.setdefault("decay_rate", 0.5)
        super().__init__(**data)
    
    def set_expiration(self, hours: float) -> None:
        """
        Set expiration time for working memory.
        
        Args:
            hours: Hours until expiration
        """
        self.expiration_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
    
    def is_expired(self) -> bool:
        """Check if working memory has expired."""
        return datetime.datetime.now() > self.expiration_time


class MemoryShell:
    """
    Memory shell architecture for agent cognitive persistence.
    
    The MemoryShell provides:
    - Multi-tiered memory system (working, episodic, semantic)
    - Configurable decay rates for different memory types
    - Time-based and access-based memory reinforcement
    - Associative memory network with activation spread
    - Query capabilities with relevance ranking
    """
    
    def __init__(self, decay_rate: float = 0.2):
        """
        Initialize memory shell.
        
        Args:
            decay_rate: Base decay rate for memories
        """
        self.memories: Dict[str, Memory] = {}
        self.decay_rate = decay_rate
        self.working_memory_capacity = 7  # Miller's number (7±2)
        self.episodic_index: Dict[str, Set[str]] = defaultdict(set)  # Tag -> memory IDs
        self.semantic_index: Dict[str, Set[str]] = defaultdict(set)  # Tag -> memory IDs
        self.temporal_sequence: List[str] = []  # Ordered list of episodic memory IDs
        self.activation_threshold = 0.1  # Minimum activation for retrieval
        
        # Initialize memory statistics
        self.stats = {
            "total_memories_created": 0,
            "total_memories_decayed": 0,
            "working_memory_count": 0,
            "episodic_memory_count": 0,
            "semantic_memory_count": 0,
            "average_salience": 0.0,
            "association_count": 0,
        }
    
    def add_working_memory(self, content: Dict[str, Any], priority: int = 1,
                         expiration_hours: float = 1.0, tags: List[str] = None) -> str:
        """
        Add item to working memory.
        
        Args:
            content: Memory content
            priority: Priority level (higher = more important)
            expiration_hours: Hours until expiration
            tags: Semantic tags
            
        Returns:
            Memory ID
        """
        # Create working memory
        memory = WorkingMemory(
            content=content,
            priority=priority,
            decay_rate=self.decay_rate * 2,  # Working memory decays faster
            tags=tags or [],
            source="working",
        )
        
        # Set expiration
        memory.set_expiration(expiration_hours)
        
        # Store in memory dictionary
        self.memories[memory.id] = memory
        
        # Add to indices
        for tag in memory.tags:
            self.episodic_index[tag].add(memory.id)
        
        # Enforce capacity limit
        self._enforce_working_memory_capacity()
        
        # Update stats
        self.stats["total_memories_created"] += 1
        self.stats["working_memory_count"] += 1
        
        return memory.id
    
    def add_episodic_memory(self, content: Dict[str, Any], emotional_valence: float = 0.0,
                         outcome: Optional[str] = None, tags: List[str] = None) -> str:
        """
        Add episodic memory.
        
        Args:
            content: Memory content
            emotional_valence: Emotional charge (-1 to 1)
            outcome: Outcome of the experience
            tags: Semantic tags
            
        Returns:
            Memory ID
        """
        # Create episodic memory
        memory = EpisodicMemory(
            content=content,
            emotional_valence=emotional_valence,
            outcome=outcome,
            decay_rate=self.decay_rate,
            tags=tags or [],
            source="episode",
        )
        
        # Set sequence position
        memory.sequence_position = len(self.temporal_sequence)
        
        # Store in memory dictionary
        self.memories[memory.id] = memory
        
        # Add to indices
        for tag in memory.tags:
            self.episodic_index[tag].add(memory.id)
        
        # Add to temporal sequence
        self.temporal_sequence.append(memory.id)
        
        # Update stats
        self.stats["total_memories_created"] += 1
        self.stats["episodic_memory_count"] += 1
        
        return memory.id
    
    def add_semantic_memory(self, content: Dict[str, Any], certainty: float = 0.7,
                        tags: List[str] = None) -> str:
        """
        Add semantic memory.
        
        Args:
            content: Memory content
            certainty: Certainty level (0-1)
            tags: Semantic tags
            
        Returns:
            Memory ID
        """
        # Create semantic memory
        memory = SemanticMemory(
            content=content,
            certainty=certainty,
            decay_rate=self.decay_rate * 0.5,  # Semantic memory decays slower
            tags=tags or [],
            source="semantic",
        )
        
        # Store in memory dictionary
        self.memories[memory.id] = memory
        
        # Add to indices
        for tag in memory.tags:
            self.semantic_index[tag].add(memory.id)
        
        # Update stats
        self.stats["total_memories_created"] += 1
        self.stats["semantic_memory_count"] += 1
        
        return memory.id
    
    def add_experience(self, experience: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Add new experience as episodic memory and extract semantic memories.
        
        Args:
            experience: Experience data
            
        Returns:
            Tuple of (episodic_id, list of semantic_ids)
        """
        # Extract tags from experience
        tags = experience.get("tags", [])
        if not tags and "type" in experience:
            tags = [experience["type"]]
        
        # Create episodic memory
        episodic_id = self.add_episodic_memory(
            content=experience,
            emotional_valence=experience.get("emotional_valence", 0.0),
            outcome=experience.get("outcome"),
            tags=tags,
        )
        
        # Extract semantic information (simple implementation)
        semantic_ids = []
        if "insights" in experience and isinstance(experience["insights"], list):
            for insight in experience["insights"]:
                if isinstance(insight, dict):
                    semantic_id = self.add_semantic_memory(
                        content=insight,
                        certainty=insight.get("confidence", 0.7),
                        tags=insight.get("tags", tags),
                    )
                    semantic_ids.append(semantic_id)
                    
                    # Create bidirectional association
                    self.add_association(episodic_id, semantic_id, 0.8)
        
        return episodic_id, semantic_ids
    
    def add_association(self, memory_id1: str, memory_id2: str, strength: float = 0.5) -> bool:
        """
        Add bidirectional association between memories.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            strength: Association strength (0-1)
            
        Returns:
            Success status
        """
        # Verify memories exist
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return False
        
        # Add bidirectional association
        self.memories[memory_id1].add_association(memory_id2, strength)
        self.memories[memory_id2].add_association(memory_id1, strength)
        
        # Update stats
        self.stats["association_count"] += 2
        
        return True
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory data or None if not found
        """
        if memory_id not in self.memories:
            return None
        
        # Get memory
        memory = self.memories[memory_id]
        
        # Update access statistics
        memory.update_access()
        
        # Convert to dictionary
        memory_dict = memory.as_dict()
        
        return memory_dict
    
    def query_memories(self, query: Dict[str, Any], memory_type: Optional[str] = None,
                    tags: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query memories based on content, type, and tags.
        
        Args:
            query: Query terms
            memory_type: Optional filter by memory type
            tags: Optional filter by tags
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        # Filter by memory type
        candidate_ids = set()
        
        if memory_type:
            # Filter by specified memory type
            for memory_id, memory in self.memories.items():
                if memory.memory_type == memory_type:
                    candidate_ids.add(memory_id)
        else:
            # Include all memory IDs
            candidate_ids = set(self.memories.keys())
        
        # Filter by tags if provided
        if tags:
            tag_memories = set()
            for tag in tags:
                # Combine episodic and semantic indices
                tag_memories.update(self.episodic_index.get(tag, set()))
                tag_memories.update(self.semantic_index.get(tag, set()))
            
            # Restrict to memories with matching tags
            if tag_memories:
                candidate_ids = candidate_ids.intersection(tag_memories)
        
        # Score candidates based on query relevance and salience
        scored_candidates = []
        
        for memory_id in candidate_ids:
            memory = self.memories[memory_id]
            
            # Skip memories below activation threshold
            current_salience = memory.calculate_current_salience()
            if current_salience < self.activation_threshold:
                continue
            
            # Calculate relevance score
            relevance = self._calculate_relevance(memory, query)
            
            # Combine relevance and salience for final score
            score = 0.7 * relevance + 0.3 * current_salience
            
            # Add to candidates
            scored_candidates.append((memory_id, score))
        
        # Sort by score (descending) and take top 'limit' results
        top_candidates = heapq.nlargest(limit, scored_candidates, key=lambda x: x[1])
        
        # Retrieve and return memories
        result_memories = []
        for memory_id, score in top_candidates:
            memory = self.memories[memory_id]
            memory.update_access()  # Update access time
            
            # Add memory with score
            memory_dict = memory.as_dict()
            memory_dict["relevance_score"] = score
            
            result_memories.append(memory_dict)
        
        return result_memories
    
    def get_recent_memories(self, memory_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get most recent memories by creation time.
        
        Args:
            memory_type: Optional filter by memory type
            limit: Maximum number of results
            
        Returns:
            List of recent memories
        """
        # Filter and sort memories by creation time
        recent_memories = []
        
        for memory_id, memory in self.memories.items():
            # Filter by memory type if specified
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Add to candidates
            recent_memories.append((memory_id, memory.creation_time))
        
        # Sort by creation time (descending)
        recent_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Retrieve and return top 'limit' memories
        result_memories = []
        for memory_id, _ in recent_memories[:limit]:
            memory = self.memories[memory_id]
            memory.update_access()  # Update access time
            result_memories.append(memory.as_dict())
        
        return result_memories
    
    def get_temporal_sequence(self, start_index: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get temporal sequence of episodic memories.
        
        Args:
            start_index: Starting index in sequence
            limit: Maximum number of results
            
        Returns:
            List of episodic memories in temporal order
        """
        # Get subset of temporal sequence
        sequence_slice = self.temporal_sequence[start_index:start_index+limit]
        
        # Retrieve and return memories
        result_memories = []
        for memory_id in sequence_slice:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.update_access()  # Update access time
                result_memories.append(memory.as_dict())
        
        return result_memories
    
    def get_relevant_experiences(self, query: Optional[Dict[str, Any]] = None, 
                              tags: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant episodic experiences.
        
        Args:
            query: Optional query terms
            tags: Optional filter by tags
            limit: Maximum number of results
            
        Returns:
            List of relevant experiences
        """
        # If query provided, use query_memories with episodic filter
        if query:
            return self.query_memories(query, memory_type="episodic", tags=tags, limit=limit)
        
        # Otherwise get most salient episodic memories
        # Filter by tags if provided
        candidate_ids = set()
        if tags:
            for tag in tags:
                candidate_ids.update(self.episodic_index.get(tag, set()))
        else:
            # Get all episodic memories
            candidate_ids = {memory_id for memory_id, memory in self.memories.items()
                          if memory.memory_type == "episodic"}
        
        # Score by current salience
        scored_candidates = []
        for memory_id in candidate_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                current_salience = memory.calculate_current_salience()
                
                # Skip memories below activation threshold
                if current_salience < self.activation_threshold:
                    continue
                
                scored_candidates.append((memory_id, current_salience))
        
        # Sort by salience (descending) and take top 'limit' results
        top_candidates = heapq.nlargest(limit, scored_candidates, key=lambda x: x[1])
        
        # Retrieve and return memories
        result_memories = []
        for memory_id, _ in top_candidates:
            memory = self.memories[memory_id]
            memory.update_access()  # Update access time
            result_memories.append(memory.as_dict())
        
        return result_memories
    
    def get_beliefs(self, tags: Optional[List[str]] = None, certainty_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get semantic beliefs with high certainty.
        
        Args:
            tags: Optional filter by tags
            certainty_threshold: Minimum certainty threshold
            
        Returns:
            List of semantic beliefs
        """
        # Filter by tags if provided
        candidate_ids = set()
        if tags:
            for tag in tags:
                candidate_ids.update(self.semantic_index.get(tag, set()))
        else:
            # Get all semantic memories
            candidate_ids = {memory_id for memory_id, memory in self.memories.items()
                          if memory.memory_type == "semantic"}
        
        # Filter and score by certainty and salience
        scored_candidates = []
        for memory_id in candidate_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                
                # Skip if not semantic or below certainty threshold
                if memory.memory_type != "semantic" or not hasattr(memory, "certainty"):
                    continue
                
                if memory.certainty < certainty_threshold:
                    continue
                
                # Calculate current salience
                current_salience = memory.calculate_current_salience()
                
                # Skip memories below activation threshold
                if current_salience < self.activation_threshold:
                    continue
                
                # Score combines certainty and salience
                score = 0.6 * memory.certainty + 0.4 * current_salience
                
                scored_candidates.append((memory_id, score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Retrieve and return memories
        result_memories = []
        for memory_id, score in scored_candidates:
            memory = self.memories[memory_id]
            memory.update_access()  # Update access time
            
            # Convert to dictionary and add score
            memory_dict = memory.as_dict()
            memory_dict["belief_score"] = score
            
            result_memories.append(memory_dict)
        
        return result_memories
    
    def apply_decay(self) -> int:
        """
        Apply memory decay to all memories and clean up decayed memories.
        
        Returns:
            Number of memories removed due to decay
        """
        # Track memories to remove
        to_remove = []
        
        # Check all memories
        for memory_id, memory in self.memories.items():
            # Calculate current salience
            current_salience = memory.calculate_current_salience()
            
            # Mark for removal if below threshold
            if current_salience < self.activation_threshold:
                to_remove.append(memory_id)
        
        # Remove decayed memories
        for memory_id in to_remove:
            self._remove_memory(memory_id)
        
        # Update stats
        self.stats["total_memories_decayed"] += len(to_remove)
        self.stats["working_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "working")
        self.stats["episodic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "episodic")
        self.stats["semantic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "semantic")
        
        # Calculate average salience
        if self.memories:
            self.stats["average_salience"] = sum(m.calculate_current_salience() for m in self.memories.values()) / len(self.memories)
        
        return len(to_remove)
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidate episodic memories into semantic memories.
        
        Returns:
            Consolidation results
        """
        # Not fully implemented - this would involve more complex semantic extraction
        # Simple implementation: just extract common tags from recent episodic memories
        recent_episodic = self.get_recent_memories(memory_type="episodic", limit=10)
        
        # Count tag occurrences
        tag_counts = defaultdict(int)
        for memory in recent_episodic:
            for tag in memory.get("tags", []):
                tag_counts[tag] += 1
        
        # Find common tags (appearing in at least 3 memories)
        common_tags = {tag for tag, count in tag_counts.items() if count >= 3}
        
        # Create semantic memory for common tags if not empty
        consolidated_ids = []
        if common_tags:
            # Create semantic memory
            semantic_id = self.add_semantic_memory(
                content={
                    "consolidated_from": [m.get("id") for m in recent_episodic],
                    "common_tags": list(common_tags),
                    "summary": f"Consolidated memory with common tags: {', '.join(common_tags)}"
                },
                certainty=0.6,
                tags=list(common_tags),
            )
            
            consolidated_ids.append(semantic_id)
        
        return {
            "consolidated_count": len(consolidated_ids),
            "consolidated_ids": consolidated_ids,
            "common_tags": list(common_tags) if common_tags else []
        }
    
    def _calculate_relevance(self, memory: Memory, query: Dict[str, Any]) -> float:
        """
        Calculate relevance score of memory to query.
        
        Args:
            memory: Memory to score
            query: Query terms
            
        Returns:
            Relevance score (0-1)
        """
        # Simple implementation: check for key overlaps
        relevance = 0.0
        
        # Extract memory content
        content = memory.content
        
        # Count matching keys at top level
        matching_keys = set(query.keys()).intersection(set(content.keys()))
        if matching_keys:
            relevance += 0.3 * (len(matching_keys) / len(query))
        
        # Check for matching values (simple string contains)
        for key, value in query.items():
            if key in content and isinstance(value, str) and isinstance(content[key], str):
                if value.lower() in content[key].lower():
                    relevance += 0.2
            elif key in content and value == content[key]:
                relevance += 0.3
        
        # Check for tag matches
        query_tags = query.get("tags", [])
        if isinstance(query_tags, list) and memory.tags:
            matching_tags = set(query_tags).intersection(set(memory.tags))
            if matching_tags:
                relevance += 0.3 * (len(matching_tags) / len(query_tags))
        
        # Cap relevance at 1.0
        return min(1.0, relevance)
    
    def _enforce_working_memory_capacity(self) -> None:
        """Enforce working memory capacity limit by removing low priority items."""
        # Count working memories
        working_memories = [(memory_id, memory) for memory_id, memory in self.memories.items()
                          if memory.memory_type == "working"]
        
        # Check if over capacity
        if len(working_memories) <= self.working_memory_capacity:
            return
        
        # Sort by priority (ascending) and salience (ascending)
        working_memories.sort(key=lambda x: (x[1].priority, x[1].calculate_current_salience()))
        
        # Remove lowest priority items until under capacity
        for memory_id, _ in working_memories[:len(working_memories) - self.working_memory_capacity]:
            self._remove_memory(memory_id)
    
    def _remove_memory(self, memory_id: str) -> None:
        """
        Remove memory by ID.
        
        Args:
            memory_id: Memory ID to remove
        """
        if memory_id not in self.memories:
            return
        
        # Get memory before removal
        memory = self.memories[memory_id]
        
        # Remove from memory dictionary
        del self.memories[memory_id]
        
        # Remove from indices
        for tag in memory.tags:
            if memory.memory_type == "episodic" and tag in self.episodic_index:
                self.episodic_index[tag].discard(memory_id)
            elif memory.memory_type == "semantic" and tag in self.semantic_index:
                self.semantic_index[tag].discard(memory_id)
        
        # Remove from temporal sequence if episodic
        if memory.memory_type == "episodic":
            if memory_id in self.temporal_sequence:
                self.temporal_sequence.remove(memory_id)
        
        # Update associations in other memories
        for other_id, other_memory in self.memories.items():
            if memory_id in other_memory.associations:
                del other_memory.associations[memory_id]
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export memory shell state.
        
        Returns:
            Serializable memory shell state
        """
        # Export memory dictionaries
        memory_dicts = {memory_id: memory.as_dict() for memory_id, memory in self.memories.items()}
        
        # Export indices (convert sets to lists for serialization)
        episodic_index = {tag: list(memories) for tag, memories in self.episodic_index.items()}
        semantic_index = {tag: list(memories) for tag, memories in self.semantic_index.items()}
        
        # Export state
        state = {
            "memories": memory_dicts,
            "episodic_index": episodic_index,
            "semantic_index": semantic_index,
            "temporal_sequence": self.temporal_sequence,
            "decay_rate": self.decay_rate,
            "activation_threshold": self.activation_threshold,
            "working_memory_capacity": self.working_memory_capacity,
            "stats": self.stats,
        }
        
        return state
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import memory shell state.
        
        Args:
            state: Memory shell state
        """
        # Clear current state
        self.memories = {}
        self.episodic_index = defaultdict(set)
        self.semantic_index = defaultdict(set)
        self.temporal_sequence = []
        
        # Import configuration
        self.decay_rate = state.get("decay_rate", self.decay_rate)
        self.activation_threshold = state.get("activation_threshold", self.activation_threshold)
        self.working_memory_capacity = state.get("working_memory_capacity", self.working_memory_capacity)
        
        # Import memories
        for memory_id, memory_dict in state.get("memories", {}).items():
            memory_type = memory_dict.get("memory_type")
            
            if memory_type == "working":
                # Create working memory
                memory = WorkingMemory(
                    id=memory_id,
                    content=memory_dict.get("content", {}),
                    priority=memory_dict.get("priority", 1),
                    decay_rate=memory_dict.get("decay_rate", self.decay_rate * 2),
                    tags=memory_dict.get("tags", []),
                    source=memory_dict.get("source", "working"),
                    salience=memory_dict.get("salience", 1.0),
                    creation_time=datetime.datetime.fromisoformat(memory_dict.get("creation_time", datetime.datetime.now().isoformat())),
                    last_access_time=datetime.datetime.fromisoformat(memory_dict.get("last_access_time", datetime.datetime.now().isoformat())),
                    access_count=memory_dict.get("access_count", 1),
                    associations=memory_dict.get("associations", {}),
                )
                
                # Set expiration time
                if "expiration_time" in memory_dict:
                    memory.expiration_time = datetime.datetime.fromisoformat(memory_dict["expiration_time"])
                else:
                    memory.set_expiration(1.0)
                
                # Store memory
                self.memories[memory_id] = memory
                
            elif memory_type == "episodic":
                # Create episodic memory
                memory = EpisodicMemory(
                    id=memory_id,
                    content=memory_dict.get("content", {}),
                    emotional_valence=memory_dict.get("emotional_valence", 0.0),
                    outcome=memory_dict.get("outcome"),
                    decay_rate=memory_dict.get("decay_rate", self.decay_rate),
                    tags=memory_dict.get("tags", []),
                    source=memory_dict.get("source", "episode"),
                    salience=memory_dict.get("salience", 1.0),
                    creation_time=datetime.datetime.fromisoformat(memory_dict.get("creation_time", datetime.datetime.now().isoformat())),
                    last_access_time=datetime.datetime.fromisoformat(memory_dict.get("last_access_time", datetime.datetime.now().isoformat())),
                    access_count=memory_dict.get("access_count", 1),
                    associations=memory_dict.get("associations", {}),
                    sequence_position=memory_dict.get("sequence_position"),
                )
                
                # Store memory
                self.memories[memory_id] = memory
                
            elif memory_type == "semantic":
                # Create semantic memory
                memory = SemanticMemory(
                    id=memory_id,
                    content=memory_dict.get("content", {}),
                    certainty=memory_dict.get("certainty", 0.7),
                    decay_rate=memory_dict.get("decay_rate", self.decay_rate * 0.5),
                    tags=memory_dict.get("tags", []),
                    source=memory_dict.get("source", "semantic"),
                    salience=memory_dict.get("salience", 1.0),
                    creation_time=datetime.datetime.fromisoformat(memory_dict.get("creation_time", datetime.datetime.now().isoformat())),
                    last_access_time=datetime.datetime.fromisoformat(memory_dict.get("last_access_time", datetime.datetime.now().isoformat())),
                    access_count=memory_dict.get("access_count", 1),
                    associations=memory_dict.get("associations", {}),
                    contradiction_ids=memory_dict.get("contradiction_ids", []),
                    supporting_evidence=memory_dict.get("supporting_evidence", []),
                )
                
                # Store memory
                self.memories[memory_id] = memory
        
        # Import indices
        for tag, memory_ids in state.get("episodic_index", {}).items():
            self.episodic_index[tag] = set(memory_ids)
        
        for tag, memory_ids in state.get("semantic_index", {}).items():
            self.semantic_index[tag] = set(memory_ids)
        
        # Import temporal sequence
        self.temporal_sequence = state.get("temporal_sequence", [])
        
        # Import stats
        self.stats = state.get("stats", self.stats.copy())
        
        # Update stats
        self.stats["working_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "working")
        self.stats["episodic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "episodic")
        self.stats["semantic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "semantic")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory shell statistics.
        
        Returns:
            Memory statistics
        """
        # Update current stats
        self.stats["working_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "working")
        self.stats["episodic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "episodic")
        self.stats["semantic_memory_count"] = sum(1 for m in self.memories.values() if m.memory_type == "semantic")
        
        # Calculate average salience
        if self.memories:
            self.stats["average_salience"] = sum(m.calculate_current_salience() for m in self.memories.values()) / len(self.memories)
        
        # Calculate additional stats
        active_memories = sum(1 for m in self.memories.values() 
                           if m.calculate_current_salience() >= self.activation_threshold)
        
        tag_stats = {
            "episodic_tags": len(self.episodic_index),
            "semantic_tags": len(self.semantic_index),
        }
        
        decay_stats = {
            "activation_threshold": self.activation_threshold,
            "active_memory_ratio": active_memories / len(self.memories) if self.memories else 0,
            "decay_rate": self.decay_rate,
        }
        
        return {
            **self.stats,
            **tag_stats,
            **decay_stats,
            "total_memories": len(self.memories),
            "active_memories": active_memories,
        }
