"""
Tests for threading utilities.
"""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aethernav_stack.core.threading_utils import LatestHolder, StoppableThread


class TestLatestHolder:
    """Tests for LatestHolder."""
    
    def test_put_and_get(self):
        """Test basic put/get."""
        holder = LatestHolder[int]()
        holder.put(42)
        assert holder.get(timeout=1.0) == 42
    
    def test_latest_value_wins(self):
        """Test that latest value replaces old."""
        holder = LatestHolder[int]()
        holder.put(1)
        holder.put(2)
        holder.put(3)
        assert holder.get(timeout=1.0) == 3
    
    def test_get_nowait_empty(self):
        """Test get_nowait returns None when empty."""
        holder = LatestHolder[int]()
        assert holder.get_nowait() is None
    
    def test_get_timeout(self):
        """Test get timeout."""
        holder = LatestHolder[int]()
        start = time.time()
        result = holder.get(timeout=0.1)
        elapsed = time.time() - start
        
        assert result is None
        assert elapsed >= 0.1
    
    def test_peek(self):
        """Test peek doesn't consume."""
        holder = LatestHolder[int]()
        holder.put(42)
        assert holder.peek() == 42
        assert holder.peek() == 42  # Still there


class TestStoppableThread:
    """Tests for StoppableThread."""
    
    def test_start_and_stop(self):
        """Test thread can start and stop."""
        iterations = []
        
        class TestThread(StoppableThread):
            def run_loop(self):
                iterations.append(1)
                self.sleep_or_stop(0.01)
        
        thread = TestThread()
        thread.start()
        time.sleep(0.1)
        thread.stop()
        thread.join(timeout=1.0)
        
        assert len(iterations) > 0
        assert not thread.is_alive()
    
    def test_sleep_or_stop(self):
        """Test sleep_or_stop returns early on stop."""
        class TestThread(StoppableThread):
            def __init__(self):
                super().__init__()
                self.slept_full = None
            
            def run_loop(self):
                self.slept_full = self.sleep_or_stop(10.0)
        
        thread = TestThread()
        thread.start()
        time.sleep(0.1)
        thread.stop()
        thread.join(timeout=1.0)
        
        assert thread.slept_full == False  # Returned early


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
