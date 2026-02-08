"""
Thread-safe data structures for pipeline communication.

Provides simple, clean abstractions for passing data between
pipeline stages with "latest frame" semantics.
"""

import threading
from typing import TypeVar, Generic, Optional
from dataclasses import dataclass
import time

T = TypeVar('T')


class LatestHolder(Generic[T]):
    """
    Thread-safe container that always holds the latest value.
    
    Writers never block. Readers get the most recent value.
    Old values are discarded (not queued).
    
    Perfect for camera frames where freshness > completeness.
    
    Usage:
        holder = LatestHolder[Frame]()
        
        # Producer thread
        holder.put(frame)  # Never blocks
        
        # Consumer thread  
        frame = holder.get()  # Gets latest, blocks if empty
        frame = holder.get_nowait()  # Returns None if empty
    """
    
    def __init__(self):
        self._value: Optional[T] = None
        self._lock = threading.Lock()
        self._event = threading.Event()
    
    def put(self, value: T) -> None:
        """Store a new value, replacing any previous value."""
        with self._lock:
            self._value = value
            self._event.set()
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Get the latest value, blocking if none available.
        
        Args:
            timeout: Max seconds to wait (None = forever)
            
        Returns:
            Latest value, or None if timeout
        """
        if self._event.wait(timeout):
            with self._lock:
                value = self._value
                self._value = None
                self._event.clear()
                return value
        return None
    
    def get_nowait(self) -> Optional[T]:
        """Get the latest value without blocking."""
        with self._lock:
            value = self._value
            self._value = None
            self._event.clear()
            return value
    
    def peek(self) -> Optional[T]:
        """View the latest value without consuming it."""
        with self._lock:
            return self._value
    
    def clear(self) -> None:
        """Discard any stored value."""
        with self._lock:
            self._value = None
            self._event.clear()


class StoppableThread(threading.Thread):
    """
    Base class for threads that can be cleanly stopped.
    
    Subclasses implement `run_loop()` which is called repeatedly
    until `stop()` is called.
    
    Usage:
        class MyThread(StoppableThread):
            def run_loop(self):
                # Do one iteration of work
                data = self.process_something()
                
        thread = MyThread(name="worker")
        thread.start()
        # ... later ...
        thread.stop()  # Signals thread to stop
        thread.join()  # Waits for thread to finish
    """
    
    def __init__(self, name: str = "StoppableThread"):
        super().__init__(name=name, daemon=True)
        self._stop_event = threading.Event()
    
    def run(self) -> None:
        """Main thread entry point."""
        try:
            self.on_start()
            while not self._stop_event.is_set():
                self.run_loop()
        except Exception as e:
            self.on_error(e)
        finally:
            self.on_stop()
    
    def run_loop(self) -> None:
        """Override this: one iteration of thread work."""
        raise NotImplementedError
    
    def on_start(self) -> None:
        """Called once when thread starts. Override for setup."""
        pass
    
    def on_stop(self) -> None:
        """Called once when thread stops. Override for cleanup."""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Called if run_loop raises an exception."""
        import traceback
        traceback.print_exc()
    
    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()
    
    @property
    def is_stopping(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()
    
    def sleep_or_stop(self, seconds: float) -> bool:
        """
        Sleep for the given duration, or return early if stopped.
        
        Returns:
            True if slept full duration, False if stop was requested
        """
        return not self._stop_event.wait(seconds)
