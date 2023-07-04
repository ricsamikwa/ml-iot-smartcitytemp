def get_rolling_window_bounds(start, end, window_size, step, iteration):
    """
    Get the start and end values of the rolling window for a given iteration index and step size.
    Returns:
    Tuple containing the start and end values of the current window
    """
    window_start = start + (iteration * step)
    window_end = window_start + window_size - 1
    return window_start, window_end

for t in range(100):
      # predictions
      start, end = get_rolling_window_bounds(0,300, 50, 2, t)

      print(start, end)