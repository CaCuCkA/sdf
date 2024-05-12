class EvaluationResult:
    __slots__ = ['model_size', 'quality_surface', 'quality_volume', 'time_per_batch', 'time_per_point']

    def __init__(self, model_size: float, quality_surface: float, quality_volume: float, time_per_batch: float, time_per_point: float):
        self.model_size = model_size
        self.quality_surface = quality_surface
        self.quality_volume = quality_volume
        self.time_per_batch = time_per_batch
        self.time_per_point = time_per_point

    def __repr__(self):
        return (f"EvaluationResult(model_size={self.model_size:.2f} MB, "
                f"quality_surface={self.quality_surface:.2f}, "
                f"quality_volume={self.quality_volume:.2f}, "
                f"time_per_batch={self.time_per_batch:.3f} ms, "
                f"time_per_point={self.time_per_point:.3f} ns)")

    def as_dict(self):
        return {field: getattr(self, field) for field in self.__slots__}
