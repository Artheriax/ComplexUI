async function convertToPng(blob: Blob): Promise<Blob> {
  if (typeof createImageBitmap !== "function") {
    throw new Error("This browser cannot convert the image to PNG.");
  }
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d");
    if (context === null) {
      throw new Error("Could not prepare the image for copying.");
    }
    context.drawImage(bitmap, 0, 0);
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result !== null) {
          resolve(result);
        } else {
          reject(new Error("Could not encode the image for copying."));
        }
      }, "image/png");
    });
  } finally {
    bitmap.close();
  }
}

async function fetchPng(url: string): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load the image (${response.status}).`);
  }
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) {
    throw new Error("The selected output is not an image.");
  }
  return blob.type === "image/png" ? blob : await convertToPng(blob);
}

export async function copyImageFromUrl(url: string): Promise<void> {
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    throw new Error("Image clipboard is not supported in this browser.");
  }
  await navigator.clipboard.write([
    new ClipboardItem({ "image/png": fetchPng(url) }),
  ]);
}
