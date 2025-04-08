from abc import ABC, abstractmethod

class Results:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, self._convert_to_results(value))

    def _convert_to_results(self, value):
        if isinstance(value, dict):
            return Results(**value)
        return value

    def __setattr__(self, key, value):
        super().__setattr__(key, self._convert_to_results(value))

    def __getattr__(self, key):
        if key not in self.__dict__:
            self.__dict__[key] = Results()
        return self.__dict__[key]

    def __getitem__(self, key):
        """Allow dictionary-like access."""
        if key not in self.__dict__:
            raise KeyError(f"Key '{key}' not found.")
        return self.__dict__[key]

    def __setitem__(self, key, value):
        """Allow dictionary-like setting."""
        self.__dict__[key] = self._convert_to_results(value)

    def set(self, key, value):
        """Dynamically set an attribute with nested support."""
        keys = key.split('.')
        target = self
        for k in keys[:-1]:
            if not hasattr(target, k) or not isinstance(getattr(target, k), Results):
                setattr(target, k, Results())
            target = getattr(target, k)
        setattr(target, keys[-1], self._convert_to_results(value))

    def get(self, key, default=None):
        """Dynamically get an attribute with nested support."""
        keys = key.split('.')
        target = self
        for k in keys:
            if not hasattr(target, k):
                return default
            target = getattr(target, k)
        return target

    def __repr__(self):
        return f"Results({self.__dict__})"

# TODO: Finish this
class AnalysisBase(ABC):
    """ Base analysis class. """

    def __init__(self, **kwargs):
        self._results = Results()

    @abstractmethod
    def run(self):
        """ Run the analysis. """
        pass

    @abstractmethod
    def save(self, filename: str):
        """ Save analysis results. """
        pass

    @property
    def results(self):
        return self._results

    # def load(self, filename: str):
    #     """ Load analysis results from a pickle file. """
    #     pass

    # @abstractmethod
    # def plot(self):
    #     """ Plots analysis results. """
    #     pass
