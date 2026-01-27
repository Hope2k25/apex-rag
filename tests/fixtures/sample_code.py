"""
Sample Python module for testing code indexing.

This module demonstrates various code structures that the
code indexer should be able to parse and extract.
"""

from typing import Optional, List
from dataclasses import dataclass


# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3


@dataclass
class User:
    """Represents a user in the system."""
    id: int
    name: str
    email: str
    is_active: bool = True


class UserService:
    """
    Service class for user operations.
    
    This class handles all user-related business logic including
    creation, retrieval, and updates.
    """

    def __init__(self, db_connection):
        """Initialize the UserService with a database connection."""
        self.db = db_connection
        self._cache = {}

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Retrieve a user by their ID.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            The User object if found, None otherwise.

        Raises:
            ValueError: If user_id is negative.
            ConnectionError: If database connection fails.
        """
        if user_id < 0:
            raise ValueError("user_id must be non-negative")
        
        if user_id in self._cache:
            return self._cache[user_id]
        
        user = self.db.fetch_user(user_id)
        if user:
            self._cache[user_id] = user
        return user

    def create_user(self, name: str, email: str) -> User:
        """
        Create a new user.

        Args:
            name: The user's display name.
            email: The user's email address.

        Returns:
            The newly created User object.

        Raises:
            ValueError: If name or email is empty.
            DuplicateEmailError: If email already exists.
        """
        if not name:
            raise ValueError("Name cannot be empty")
        if not email:
            raise ValueError("Email cannot be empty")
        
        user = User(
            id=self.db.next_id(),
            name=name,
            email=email,
        )
        self.db.insert_user(user)
        return user

    def list_users(self, limit: int = 100) -> List[User]:
        """
        List all users with optional limit.

        Args:
            limit: Maximum number of users to return.

        Returns:
            List of User objects.
        """
        return self.db.fetch_users(limit=limit)


def validate_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: The email address to validate.

    Returns:
        True if the email format is valid, False otherwise.
    """
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))


async def async_fetch_user(user_id: int) -> Optional[User]:
    """
    Asynchronously fetch a user by ID.

    This is an async version of get_user for use in
    async contexts like FastAPI endpoints.

    Args:
        user_id: The user's unique identifier.

    Returns:
        The User object if found, None otherwise.
    """
    # Simulated async operation
    import asyncio
    await asyncio.sleep(0.1)
    return None  # Placeholder
