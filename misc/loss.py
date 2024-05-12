def mape_loss(pred, target):
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    return (difference * scale).mean()
