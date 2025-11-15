class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Supports efficient union and find operations with path compression
    and union by rank optimizations.
    """

    def __init__(self, n):
        """
        Initialize Union-Find with n elements (0 to n-1).

        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        """
        Find the root/representative of the set containing x.
        Uses path compression for optimization.

        Args:
            x: Element to find

        Returns:
            Root of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Unite the sets containing x and y.
        Uses union by rank for optimization.

        Args:
            x: Element in first set
            y: Element in second set

        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1

        return True

    def connected(self, x, y):
        """
        Check if x and y are in the same set.

        Args:
            x: First element
            y: Second element

        Returns:
            True if x and y are in the same set, False otherwise
        """
        return self.find(x) == self.find(y)

    def get_size(self, x):
        """
        Get the size of the set containing x.

        Args:
            x: Element to query

        Returns:
            Size of the set containing x
        """
        return self.size[self.find(x)]

    def count_sets(self):
        """
        Count the number of disjoint sets.

        Returns:
            Number of distinct sets
        """
        return len(set(self.find(i) for i in range(len(self.parent))))
