from dataclasses import dataclass, field
import math
from typing import Any

from minmaxheap import MinMaxHeap

@dataclass(order=True)
class PQueueItem:
    """Wrapper for priority queue items

    Items should only be compared based on their priority, not data.

    Attributes:
        priority (int):
            The priority of the item
        data (Any):
            The associated data
    """
    priority: int
    data: Any=field(compare=False)
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data
    def unwrapped(self):
        """Return the underlying (priority, data) tuple"""
        return self.priority, self.data

class PriorityQueue():
    """Double-ended priority queue backed by a min-max heap

    Args:
        items (list, optional):
            An initial list of (priority, data) tuples
        maxlen (int, optional):
            The maximum number of items
        mode (str, ['min', 'max'])
            When maxlen is reached, which priority items should continue to be saved
    """
    def __init__(self, items=None, maxlen=None, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("mode must be either 'min' or 'max'")
        self.mode = mode
        if maxlen is not None:
            if not isinstance(maxlen, int):
                raise TypeError('Expected maxlen to be of type int')
        self.maxlen = maxlen
        if items is not None:
            if not isinstance(items[0], tuple):
                raise TypeError('PriorityQueue expects a list of tuples')
            items = [PQueueItem(*item) for item in items]
            self.heap = MinMaxHeap.from_items(items)
        else:
            self.heap = MinMaxHeap()

        if self.maxlen is not None:
            while len(self.heap) > maxlen:
                self._eject_one()
        self.n_items = len(self.heap)
        self.lookup = {item.data: item for item in self.heap}
        self.REMOVED = '<removed>'

    def __len__(self):
        return self.n_items

    def push(self, item, priority):
        """Add a new item or update the priority of an existing item"""
        if item in self.lookup:
            self.remove(item)
        item = PQueueItem(priority, item)
        if self.maxlen is not None and len(self) >= self.maxlen:
            should_eject = (
                (self.mode == 'max' and self._peek_min().priority < item.priority)
                or (self.mode == 'min' and self._peek_max().priority > item.priority)
            )
            if should_eject:
                self._eject_one()
            else:
                return # not enough room; drop this item
        else:
            self.n_items += 1
        self.heap.push(item)
        self.lookup[item.data] = item

    def remove(self, item):
        """Mark an existing item as REMOVED.  Raise KeyError if not found."""
        entry = self.lookup.pop(item)
        entry.data = self.REMOVED
        self.n_items -= 1

    def peek_min(self):
        """Return the lowest priority item without removing it"""
        return self._peek_min().data

    def _peek_min(self):
        """Return the lowest priority PQItem without removing it"""
        while self.heap:
            item = self.heap.peek_min()
            if item.data is not self.REMOVED:
                return item
            self.heap.pop_min()
        raise KeyError('peek called on empty priority queue')

    def peek_max(self):
        """Return the highest priority item without removing it"""
        return self._peek_max().data

    def _peek_max(self):
        """Return the highest priority PQItem without removing it"""
        while self.heap:
            item = self.heap.peek_max()
            if item.data is not self.REMOVED:
                return item
            self.heap.pop_max()
        raise KeyError('peek called on empty priority queue')

    def pop_min(self):
        """Remove and return the lowest priority item"""
        while self.heap:
            item = self.heap.pop_min()
            if item.data is not self.REMOVED:
                self.n_items -= 1
                del self.lookup[item.data]
                return item.data
        raise KeyError('pop called on empty priority queue')

    def pop_max(self):
        """Remove and return the highest priority item"""
        while self.heap:
            item = self.heap.pop_max()
            if item.data is not self.REMOVED:
                self.n_items -= 1
                del self.lookup[item.data]
                return item.data
        raise KeyError('pop called on empty priority queue')

    def _eject_one(self):
        if self.mode == 'min':
            item = self.heap.pop_max()
        else:
            item = self.heap.pop_min()
        return item

    def __iter__(self):
        """Return the list of (priority, data) tuples in the queue"""
        return iter(item.data for item in self.heap if item.data is not self.REMOVED)



def test_basics():
    # Basic types
    queue = PriorityQueue()
    assert not queue
    queue.push('foo', 10)
    queue.push('bar', 9)
    queue.push('baz', 11)
    assert queue
    assert queue.peek_min() == 'bar'
    assert queue.peek_max() == 'baz'
    assert queue.pop_min() == 'bar'
    assert len(queue) == 2
    assert queue.pop_max() == 'baz'
    assert queue.peek_min() == queue.peek_max() == 'foo'
    del queue
    print('Test passed: basic types')

def test_maxlen():
    queue = PriorityQueue(maxlen=3, mode='max')
    queue.push('foo', 10)
    queue.push('bar', 7)
    queue.push('baz', 15)
    queue.push('fiz', 9)
    queue.push('buz', 12)
    # assert list(iter(queue)) == ['foo', 'baz', 'buz']
    assert len(queue) == 3
    assert queue.pop_min() == 'foo'
    assert queue.pop_max() == 'baz'

    queue = PriorityQueue(maxlen=3, mode='min')
    queue.push('foo', 10)
    queue.push('bar', 7)
    queue.push('baz', 15)
    queue.push('fiz', 9)
    queue.push('buz', 12)
    # assert queue.items() == [(7, 1, 'bar'), (9, 3, 'fiz'), (10, 0, 'foo')]
    assert len(queue) == 3
    assert queue.pop_min() == 'bar'
    assert queue.pop_max() == 'foo'
    print('Test passed: maxlen')

def test_many_elements():
    queue = PriorityQueue()
    for i in range(20):
        queue.push(i, priority=i)
    for i in range(20):
        item = queue.pop_min()
        assert item == i, '{} != {}'.format(item,i)

    queue = PriorityQueue()
    for i in range(20):
        queue.push(i, priority=i)
    for i in range(20):
        item = queue.pop_max()
        assert item == 19-i, '{} != {}'.format(item,19-i)

    print('Test passed: many elements')

def test_same_priority():
    queue = PriorityQueue()
    queue.push('first', priority=2)
    for i in range(20):
        queue.push(i, priority=1)
    queue.push('last', priority=0)

    assert queue.pop_max() == 'first'
    assert queue.pop_min() == 'last'
    assert len(queue) == 20

    print('Test passed: same priority')

def test():
    """Test priority queue functionality"""

    test_basics()
    test_maxlen()
    test_many_elements()
    test_same_priority()

    print('All tests passed.')

if __name__ == '__main__':
    test()
