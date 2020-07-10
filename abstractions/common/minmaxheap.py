import math

class MinMaxHeap:
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return iter(self.heap)

    @classmethod
    def from_items(cls, items):
        heap = cls()
        heap.heap = list(items)
        first_half = reversed(range(len(items)//2))
        for i in first_half:
            heap._push_down(i)
        return heap

    def push(self, item):
        i = len(self)
        self.heap.append(item)
        self._push_up(i)

    def peek_min(self):
        return self.heap[self._find_min()]

    def peek_max(self):
        return self.heap[self._find_max()]

    def pop_min(self):
        min_index = self._find_min()
        return self._pop(min_index)

    def pop_max(self):
        max_index = self._find_max()
        return self._pop(max_index)

    def _pop(self, i):
        item = self.heap[i]
        self.heap[i], self.heap[-1] = self.heap[-1], self.heap[i] # swap
        del self.heap[-1]
        self._push_down(i)
        return item

    def _find_min(self):
        if self.heap:
            return 0
        raise KeyError('find_min called on heap with no elements')

    def _find_max(self):
        if self.heap:
            root = 0
            candidates = [root] + self._get_children(root)
            return max([(self.heap[idx], idx) for idx in candidates])[1]
        raise KeyError('find_max called on heap with no elements')

    @staticmethod
    def _is_on_min_level(i):
        level = int(math.floor(math.log2(i+1)))
        return level % 2 == 0

    @staticmethod
    def _is_root(i):
        return i==0

    @staticmethod
    def _is_child_of(i, parent):
        return i in [2*parent+1, 2*parent+2]

    @staticmethod
    def _is_grandchild_of(i, grandparent):
        return 4*grandparent+3 <= i <= 4*grandparent+6

    @staticmethod
    def _has_parent(self, i):
        return i>0

    @staticmethod
    def _get_parent(i):
        if i == 0:
            return None
        return (i-1)//2

    @staticmethod
    def _has_grandparent(i):
        return i>2

    @staticmethod
    def _get_grandparent(i):
        return ((i-1)//2 - 1)//2

    def _has_children(self, i):
        return len(self) > 2*i+1

    def _get_children(self, i):
        left = 2*i+1
        right = 2*i+2
        children = []
        if len(self) > left:
            children.append(left)
        if len(self) > right:
            children.append(right)
        return children

    def _has_grandchildren(self, i):
        return len(self) > 4*i+3

    def _get_grandchildren(self, i):
        children = self._get_children(i)
        grandchildren = []
        for child in children:
            left = 2*child+1
            right = 2*child+2
            if len(self) > left:
                grandchildren.append(left)
            if len(self) > right:
                grandchildren.append(right)
        return grandchildren

    def _push_down(self, i):
        if self._is_on_min_level(i):
            self._push_down_min(i)
        else:
            self._push_down_max(i)

    def _push_down_min(self, m):
        self._push_down_iter(m, swap_idx=min, swap_cond=lambda a, b: a < b)

    def _push_down_max(self, m):
        self._push_down_iter(m, swap_idx=max, swap_cond=lambda a, b: a > b)

    def _push_down_iter(self, m, swap_idx, swap_cond):
        while self._has_children(m):
            i = m
            successors = self._get_children(i) + self._get_grandchildren(i)
            m = swap_idx([(self.heap[idx], idx) for idx in successors])[1]
            if swap_cond(self.heap[m], self.heap[i]):
                self.heap[m], self.heap[i] = self.heap[i], self.heap[m] #swap
                if self._is_grandchild_of(m, grandparent=i):
                    p = self._get_parent(m)
                    if swap_cond(self.heap[p], self.heap[m]):
                        self.heap[m], self.heap[p] = self.heap[p], self.heap[m] #swap
                    continue
            break

    def _push_up(self, i):
        if not self._is_root(i):
            p = self._get_parent(i)
            if self._is_on_min_level(i):
                if self.heap[i] > self.heap[p]:
                    self.heap[i], self.heap[p] = self.heap[p], self.heap[i] #swap
                    self._push_up_max(p)
                else:
                    self._push_up_min(i)
            else:
                if self.heap[i] < self.heap[p]:
                    self.heap[i], self.heap[p] = self.heap[p], self.heap[i] #swap
                    self._push_up_min(p)
                else:
                    self._push_up_max(i)

    def _push_up_min(self, i):
        self._push_up_iter(i, swap_cond=lambda a, b: a < b)

    def _push_up_max(self, i):
        self._push_up_iter(i, swap_cond=lambda a, b: a > b)

    def _push_up_iter(self, i, swap_cond):
        while self._has_grandparent(i):
            gp = self._get_grandparent(i)
            if swap_cond(self.heap[i], self.heap[gp]):
                self.heap[i], self.heap[gp] = self.heap[gp], self.heap[i] #swap
                i = gp
            else:
                break

    def _is_valid(self):
        for position, item in enumerate(self.heap):
            if self._is_on_min_level(position):
                decendants = self._get_children(position)
                while decendants:
                    decendant = decendants.pop()
                    if decendant < item:
                        return False
                    decendants += self._get_children(decendant)
            else:
                while decendants:
                    decendant = decendants.pop()
                    if decendant > item:
                        return False
                    decendants += self._get_children(decendant)
        return True

def test():
    # Test push
    heap_a = MinMaxHeap()
    for i in range(6):
        heap_a.push(i)
    heap_a.push(6)
    assert heap_a._is_valid()

    # Test from_items
    heap_b = MinMaxHeap.from_items(range(7))
    assert heap_b._is_valid()

    import copy
    heap_c = copy.deepcopy(heap_a)
    heap_d = copy.deepcopy(heap_b)

    # Test pop_min
    assert len(heap_b) == len(heap_a)
    while heap_b:
        assert heap_a.peek_min() == heap_b.peek_min()
        assert heap_a.peek_max() == heap_b.peek_max()
        item_a = heap_a.pop_min()
        item_b = heap_b.pop_min()
        assert item_a == item_b

    # Test pop_max
    assert len(heap_d) == len(heap_c)
    while heap_d:
        assert heap_c.peek_min() == heap_d.peek_min()
        assert heap_c.peek_max() == heap_d.peek_max()
        item_c = heap_c.pop_max()
        item_d = heap_d.pop_max()
        print(item_c, item_d)
        assert item_c == item_d

    print('All tests passed.')

if __name__ == "__main__":
    test()