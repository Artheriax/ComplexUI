# /// script
# dependencies = ["Pillow>=11.0"]
# ///
images: IMAGE_BATCH
index: int = 0

if not isinstance(images, (list, tuple)) or not images:
    raise ValueError("images must be a non-empty image batch")
if index < 0 or index >= len(images):
    raise IndexError(f"index {index} is outside the batch of {len(images)} images")

image = images[index]

return image: IMAGE
