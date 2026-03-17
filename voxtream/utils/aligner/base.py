from abc import abstractmethod


class BaseAligner:
    @abstractmethod
    def align(self, *args, **kwargs):
        """
        Perform forced alignment.
        """
        pass
