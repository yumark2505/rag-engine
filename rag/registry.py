from typing import Dict, Type, Optional


class Registry:
    def __init__(self):
        self._loaders: Dict[str, Type] = {}
        self._chunkers: Dict[str, Type] = {}
        self._embedders: Dict[str, Type] = {}
        self._vector_stores: Dict[str, Type] = {}
        self._retrievers: Dict[str, Type] = {}
        self._prompts: Dict[str, Type] = {}
        self._generators: Dict[str, Type] = {}
        
    # ── Decorator Factories ──────────────────────────────────
    def loader(self, name: str):
        def decorator(cls: Type):
            self._loaders[name] = cls
            return cls
        return decorator

    def chunker(self, name: str):
        def decorator(cls: Type):
            self._chunkers[name] = cls
            return cls
        return decorator

    def embedder(self, name: str):
        def decorator(cls: Type):
            self._embedders[name] = cls
            return cls
        return decorator

    def vector_store(self, name: str):
        def decorator(cls: Type):
            self._vector_stores[name] = cls
            return cls
        return decorator

    def retriever(self, name: str):
        def decorator(cls: Type):
            self._retrievers[name] = cls
            return cls
        return decorator

    def prompt(self, name: str):
        def decorator(cls: Type):
            self._prompts[name] = cls
            return cls
        return decorator

    def generator(self, name: str):
        def decorator(cls: Type):
            self._generators[name] = cls
            return cls
        return decorator

    # ── Getters ──────────────────────────────────────────────
    def get_loader(self, name: str) -> Optional[Type]:
        return self._loaders.get(name)

    def get_chunker(self, name: str) -> Optional[Type]:
        return self._chunkers.get(name)

    def get_embedder(self, name: str) -> Optional[Type]:
        return self._embedders.get(name)

    def get_vector_store(self, name: str) -> Optional[Type]:
        return self._vector_stores.get(name)

    def get_retriever(self, name: str) -> Optional[Type]:
        return self._retrievers.get(name)

    def get_prompt(self, name: str) -> Optional[Type]:
        return self._prompts.get(name)

    def get_generator(self, name: str) -> Optional[Type]:
        return self._generators.get(name)

    # ── List all registered ──────────────────────────────────
    @property
    def loaders(self) -> Dict[str, Type]:
        return self._loaders

    @property
    def chunkers(self) -> Dict[str, Type]:
        return self._chunkers

    @property
    def embedders(self) -> Dict[str, Type]:
        return self._embedders

    @property
    def vector_stores(self) -> Dict[str, Type]:
        return self._vector_stores

    @property
    def retrievers(self) -> Dict[str, Type]:
        return self._retrievers

    @property
    def prompts(self) -> Dict[str, Type]:
        return self._prompts

    @property
    def generators(self) -> Dict[str, Type]:
        return self._generators


registry = Registry()
