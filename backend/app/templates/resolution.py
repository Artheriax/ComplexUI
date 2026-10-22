width: int = 512
height: int = 512

if width < 8 or height < 8:
    raise ValueError("width and height must be at least 8 pixels")
if width > 16384 or height > 16384:
    raise ValueError("width and height must be at most 16384 pixels")
if width % 8 != 0 or height % 8 != 0:
    raise ValueError("width and height must be multiples of 8")

return width: INT, height: INT
