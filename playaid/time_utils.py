def timestamp_to_frame(timestamp: str):
    minutes, seconds = timestamp.split(":")
    minutes = int(minutes)
    seconds = int(seconds)
    return (seconds * 60) + (minutes * 60 * 60)


def frame_to_timestamp(frame_number, fps=60):
    total_seconds = frame_number // fps
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"


def frame_to_seconds(frame_number, fps=60):
    total_seconds = frame_number // fps
    return total_seconds
